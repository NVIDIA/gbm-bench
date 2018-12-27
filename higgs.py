# MIT License
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

from __future__ import print_function

import os
import time
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from utils import *


def prepare(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder,
                              "higgs" + "" if nrows is None else str(nrows) + ".pkl")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    higgs = pd.read_csv(local_url, nrows=nrows)
    X = higgs.iloc[:, 1:].values
    y = higgs.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)


def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 100

xgb_common_params = {
    "gamma": 0.1,
    "learning_rate": 0.1,
    "max_depth": 5,
    "max_leaves": 2 ** 5,
    "min_child_weight": 1,
    "num_round": nTrees,
    "reg_lambda": 1,
    "scale_pos_weight": 2,
    "subsample": 1,
}

lgb_common_params = {
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "min_split_gain": 0.1,
    "num_leaves": 2 ** 5,
    "num_round": nTrees,
    "objective": "binary",
    "reg_lambda": 1,
    "scale_pos_weight": 2,
    "subsample": 1,
    "task": "train",
}

cat_common_params = {
    "depth": 5,
    "iterations": nTrees,
    "l2_leaf_reg": 0.1,
    "learning_rate": 0.1,
    "loss_function": "Logloss",
}

# NOTES: some benchmarks are disabled!
#  . xgb-cpu takes almost an hour to train
#  . xgb-gpu runs out of memory
benchmarks = {
    "xgb-cpu-exact": (False, XgbBenchmark, metrics,
                      dict(xgb_common_params, tree_method="exact",
                           nthread=nthreads)),
    "xgb-cpu": (True, XgbBenchmark, metrics,
                dict(xgb_common_params, nthread=nthreads,
                     grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu-exact": (False, XgbBenchmark, metrics,
                      dict(xgb_common_params, tree_method="gpu_exact",
                           objective="gpu:binary:logistic")),
    "xgb-gpu": (True, XgbBenchmark, metrics,
                dict(xgb_common_params, tree_method="gpu_hist",
                     objective="gpu:binary:logistic")),
    "xgb-dask-gpu": (True, XgbDaskBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "lgbm-cpu": (True, LgbBenchmark, metrics,
                 dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu": (True, LgbBenchmark, metrics,
                 dict(lgb_common_params, device="gpu")),

    "cat-cpu": (True, CatBenchmark, catMetrics,
                dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu": (True, CatBenchmark, catMetrics,
                dict(cat_common_params, task_type="GPU")),
}

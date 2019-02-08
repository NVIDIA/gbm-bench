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
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder,
                              "year" + ("" if nrows is None else "-" + str(nrows)) + ".pkl")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    year = pd.read_csv(local_url, nrows=nrows, header=None)
    X = year.iloc[:, 1:]
    y = year.iloc[:, 0]
    # this dataset requires a specific train/test split,
    # with the specified number of rows at the start belonging to the train set,
    # and the rest being the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
                                                        train_size=463715,
                                                        test_size=51630)
    data = Data(X_train, X_test, y_train, y_test)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def metrics(y_test, y_pred):
    return regression_metrics(y_test, y_pred)


def catMetrics(y_test, y_pred):
    return regression_metrics(y_test, y_pred)


nthreads = get_number_processors()
nTrees = 100

xgb_common_params = {
    "gamma": 0.1,
    "learning_rate": 0.1,
    "max_depth": 5,
    "max_leaves": 0,
    "min_child_weight": 1,
    "num_round": nTrees,
    "objective": "reg:linear",
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "subsample": 1,
}

lgb_common_params = {
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "min_split_gain": 0.1,
    "num_leaves": 2 ** 5,
    "num_round": nTrees,
    "objective": "regression",
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "subsample": 1,
    "task": "train",
}

cat_common_params = {
    "depth": 5,
    "iterations": nTrees,
    "l2_leaf_reg": 0.1,
    "learning_rate": 0.1,
    "loss_function": "RMSE",
}

rf_common_params = dict(xgb_common_params)
rf_common_params.update({
    "colsample_bynode":  0.8,
    "learning_rate":     1.0,
    "num_parallel_tree": nTrees,
    "num_round":         1,
    "reg_lambda":        0.01,
    "random_state":      42,
    "scale_pos_weight":  1,
    "subsample":         0.8,
})

skl_rf_params = {
    "criterion": "mse",
    "max_depth": 5,
    "max_features": 0.8,
    "max_leaf_nodes": None,
    "min_samples_leaf": 1,
    "n_estimators": nTrees,
    "n_jobs": nthreads,
    "random_state": 42,
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
                      dict(xgb_common_params, tree_method="gpu_exact")),
    "xgb-gpu": (True, XgbBenchmark, metrics,
                dict(xgb_common_params, tree_method="gpu_hist")),
    "xgb-dask-gpu": (True, XgbDaskBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist")),
    "xgb-rf-gpu-exact":  (True, XgbBenchmark, metrics,
                      dict(rf_common_params, tree_method="gpu_exact")),
    "xgb-rf-gpu":        (True, XgbBenchmark, metrics,
                      dict(rf_common_params, tree_method="gpu_hist")),
    "skl-rf":        (True, SklRfRegressionBenchmark, metrics, skl_rf_params),

    "lgbm-cpu": (True, LgbBenchmark, metrics,
                 dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu": (True, LgbBenchmark, metrics,
                 dict(lgb_common_params, device="gpu")),

    "cat-cpu": (True, CatBenchmark, catMetrics,
                dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu": (True, CatBenchmark, catMetrics,
                dict(cat_common_params, task_type="GPU")),
}

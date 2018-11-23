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
from sklearn.model_selection import train_test_split

from utils import *


def generate_feables(df):
    X = df[df.columns.difference(["boson"])]
    y = df["boson"]
    return X, y

def prepareImpl(dbFolder, test_size, shuffle):
    # reading compressed csv is supported in pandas
    csv_name = "HIGGS.csv.gz"
    pkl_name = csv_name + '.pkl'
    cols = [
        "boson", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude",
        "missing_energy_phi", "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
        "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag", "jet_3_pt", "jet_3_eta",
        "jet_3_phi", "jet_3_b-tag", "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
        "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
    ]
    csv_file = os.path.join(dbFolder, csv_name)
    pkl_file = os.path.join(dbFolder, pkl_name)
    start = time.time()
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_csv(csv_file, names=cols, dtype=np.float32)
        df.to_pickle(pkl_file)
    X, y = generate_feables(df)
    strati = y if shuffle else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=strati,
                                                        random_state=77,
                                                        test_size=test_size,
                                                        shuffle=shuffle)
    load_time = time.time() - start
    print("Higgs dataset loaded in %.2fs" % load_time)
    return Data(X_train, X_test, y_train, y_test)


def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 500000, True)

def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 100

xgb_common_params = {
    "gamma":            0.1,
    "learning_rate":    0.1,
    "max_depth":        5,
    "max_leaves":       2**5,
    "min_child_weight": 1,
    "num_round":        nTrees,
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "min_child_weight": 1,
    "min_split_gain":   0.1,
    "num_leaves":       2**5,
    "num_round":        nTrees,
    "objective":        "binary",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            5,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
}

# NOTES: some benchmarks are disabled!
#  . xgb-cpu takes almost an hour to train
#  . xgb-gpu runs out of memory
benchmarks = {
    "xgb-cpu-exact":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu-exact":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gpu": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "xgb-cudf-exact":      (False, XgbCudfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-cudf": (True, XgbCudfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "xgb-dask-gpu": (True, XgbDaskBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "xgb-dask-cudf": (True, XgbDaskCudfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, catMetrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (True, CatBenchmark, catMetrics,
                     dict(cat_common_params, task_type="GPU")),
}

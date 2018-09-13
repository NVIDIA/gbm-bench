# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conversion import *
from metrics import *
from utils import *

import glob

def load_npy_files(globpath):
    files = glob.glob(globpath)

    # cheap way to shuffle the data a bit
    np.random.shuffle(files)

    # print the total number of files...
    print("Imported {} files.".format(len(files)))

    # faster way of loading/concatenating files...
    ndarr_list = []
    for f in files:
        ndarr = np.load(f)
        ndarr_list.append(ndarr)
    ndarrs = np.concatenate(ndarr_list)

    # double-check the shape...
    print("Shape of imported ndarray: {}".format(ndarrs.shape))

    # double-check first row of data...
    print("First row of data...")
    print(ndarrs[0])

    return ndarrs

def prepareImplCommon(dbFolder, testSize, shuffle, dbSubFolder, numRows):
    start = time.time()
    s = time.time()
    print("Loading npy files...")
    data = load_npy_files(os.path.join(dbFolder, dbSubFolder, "day_[0-1]/*.npy"))

    X = data[:, 1:]
    y = data[:, 0]
    del data

    print("Dataset has " + str(len(X[0])) + " input features.")
    print("Dataset has " + str(len(X)) + " rows.")

    idx = np.random.choice(np.arange(len(y)), numRows, replace=False)
    X = X[idx]
    y = y[idx]
    print("Done loading npy files. %.2fs" % (time.time() - s))
    
    s = time.time()
    print("Generating train/test split...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=shuffle, random_state=42, test_size=testSize)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None, shuffle=False, random_state=42, test_size=testSize)
    print("Done generating train/test split. %.2fs" % (time.time() - s))
    del X, y
    load_time = time.time() - start
    print("Criteo CTR dataset loaded in %.2fs" % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)

def prepareImpl(dbFolder, testSize, shuffle, nrows):
    rows = 2e7 if nrows is None else nrows
    return prepareImplCommon(dbFolder, testSize, shuffle, "etled", rows)

def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 0.01, True, nrows)


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 200

xgb_common_params = {
    "eta":              0.2,
    "gamma":            0.4,
    # "learning_rate":    0.1,
    "max_depth":        7,
    # "max_leaves":       2**8,
    "min_child_weight": 20,
    "num_round":        nTrees,
    # "reg_lambda":       1,
    # "scale_pos_weight": 2,
    "subsample":        1,
    "lambda":           100,
    "eval_metric":      "logloss",
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "alpha":            3,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "min_child_weight": 30,
    "min_split_gain":   0.1,
    "num_leaves":       2**8,
    "num_round":        nTrees,
    "objective":        "binary",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            8,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
}

# NOTES: some benchmarks are disabled!
#  . xgb-gpu  encounters illegal memory access
#[16:16:33] /xgboost/dmlc-core/include/dmlc/./logging.h:300: [16:16:33] /xgboost/src/tree/updater_gpu.cu:528: GPU plugin exception: /xgboost/src/tree/../common/device_helpers.cuh(319): an illegal memory access was encountered
#  . cat-gpu  currently segfaults
benchmarks = {
    "xgb-cpu-exact":      (True, XgbBenchmark, metrics,
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
    "xgb-gdf-exact":      (False, XgbGdfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gdf": (True, XgbGdfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, catMetrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (False, CatBenchmark, catMetrics,
                     dict(cat_common_params, task_type="GPU")),
}

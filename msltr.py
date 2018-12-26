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
import time
import sys
import subprocess

import numpy as np
from sklearn.datasets import load_svmlight_file

from utils import *



nFeatures = 137
nClasses = 5

def loadLibsvmData(datafile, nfeats):
    (X, y) = load_svmlight_file(datafile, n_features=nfeats, dtype=np.float32,
                                zero_based=True)
    X = X.toarray()
    X = X[:,1:]
    print("X: ", X.shape)
    print("y: ", y.shape)
    return (X, y)

def load_full(dbFolder, zip_name):
    dataFolder = os.path.join(dbFolder, "Fold1")
    trainTxt = os.path.join(dataFolder, "train.txt")
    testTxt = os.path.join(dataFolder, "vali.txt")
    start = time.time()
    if not os.path.exists(dataFolder):
        print("Unzipping data...")
        subprocess.check_call("cd %s && unzip -o %s" % (dbFolder, zip_name),
                              shell=True)
    (X_train, y_train) = loadLibsvmData(trainTxt, nFeatures)
    (X_test, y_test) = loadLibsvmData(testTxt, nFeatures)
    load_time = time.time() - start
    print("msltr dataset loaded and preprocessed in %.2fs" % load_time)
    return Data(X_train, X_test, y_train, y_test)

def prepare(dbFolder, nrows):
    return load_full(dbFolder, "MSLR-WEB10K.zip")

labels = [0, 1, 2, 3, 4]

def metrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_multilabel(y_test, pred, labels)

nthreads = get_number_processors()
nTrees = 100

xgb_common_params = {
    "eval_metric":      "merror",
    "gamma":            0.1,
    "learning_rate":    0.2,
    "max_depth":        8,
    "max_leaves":       2**8,
    "num_class":        len(labels),
    "min_child_weight": 1,
    "num_round":        nTrees,
    "objective":        "multi:softprob",
    "reg_lambda":       1,
    "subsample":        1,
}

lgb_common_params = {
    "learning_rate":    0.2,
    "metric":           "multi_error",
    "min_child_weight": 1,
    "min_split_gain":   0.1,
    "num_leaves":       2**8,
    "num_round":        nTrees,
    "num_class":        len(labels),
    "objective":        "multiclass",
    "reg_lambda":       1,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            8,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.2,
    "loss_function":    "Logloss",
}


benchmarks = {
    "xgb-cpu-exact":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu-exact":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact")),
    "xgb-gpu": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist")),
    "xgb-cudf-exact":      (True, XgbCudfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact")),
    "xgb-cudf": (True, XgbCudfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist")),
    "xgb-dask-gpu": (True, XgbDaskBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, metrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (True, CatBenchmark, metrics,
                     dict(cat_common_params, task_type="GPU")),
}

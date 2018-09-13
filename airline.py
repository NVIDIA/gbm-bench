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
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conversion import *
from metrics import *
from utils import *


def generate_feables(df):
    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]
    return X,y

def prepareImplCommon(dbFolder, testSize, shuffle, datasetFileName, numRows):
    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16, and load only 2e7 rows (1/6 of the dataset) to avoid swapping
    dtype = np.int16
    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance": dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }
    start = time.time()
    pklFile = os.path.join(dbFolder, "%s-%d.pkl" % (datasetFileName, numRows))
    if os.path.exists(pklFile):
        df = pd.read_pickle(pklFile)
    else:
        df1 =  pd.read_csv(os.path.join(dbFolder, datasetFileName),
                           names=cols, dtype=dtype_columns, nrows=numRows)
        df2 = convert_related_cols_categorical_to_numeric(df1, col_list=["Origin",
                                                                         "Dest"])
        del df1
        df3 = convert_cols_categorical_to_numeric(df2, col_list="UniqueCarrier")
        del df2
        df = df3
        df.to_pickle(pklFile)
    df["ArrDelayBinary"] = 1*(df["ArrDelay"] > 0)    
    X, y = generate_feables(df)
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        shuffle=shuffle,
                                                        random_state=42,
                                                        test_size=testSize)
    del X, y
    load_time = time.time() - start
    print("Airline dataset loaded in %.2fs" % load_time, file=sys.stderr)    
    return Data(X_train, X_test, y_train, y_test)

def prepareImpl(dbFolder, testSize, shuffle, nrows):
    rows = 2e7 if nrows is None else nrows
    return prepareImplCommon(dbFolder, testSize, shuffle,
                             "airline_14col.data.bz2", rows)

def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 0.2, True, nrows)


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 100


xgb_common_params = {
    "eta":               0.1,
    "gamma":             0.1,
    "learning_rate":     0.1,
    "max_depth":         8,
    "max_leaves":        2**8,
    "min_child_weight":  30,
    "num_round":         nTrees,
    "reg_lambda":        1,
    "scale_pos_weight":  2,
    "subsample":         1,
}

lgb_common_params = {
    "learning_rate":     0.1,
    "min_child_weight":  30,
    "min_split_gain":    0.1,
    "num_leaves":        2**8,
    "num_round":         nTrees,
    "objective":         "binary",
    "reg_lambda":        1,
    "scale_pos_weight":  2,
    "subsample":         1,
    "task":              "train",
}

cat_common_params = {
    "depth":             8,
    "iterations":        nTrees,
    "l2_leaf_reg":       0.1,
    "learning_rate":     0.1,
    "loss_function":     "Logloss",
}

rf_common_params = dict(xgb_common_params)
rf_common_params.update({
    "colsample_bytree":  0.8,
    "eta":               1.0,
    "num_parallel_tree": nTrees,
    "num_round":         1,
    "random_state":      42,
    "subsample":         0.8,
})


benchmarks = {
    "xgb-cpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "xgb-gdf":      (True, XgbGdfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gdf-hist": (True, XgbGdfBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),

    "rf-gpu-hist":  (True, XgbBenchmark, metrics,
                     dict(rf_common_params, tree_method="gpu_hist",
                          objective="gpu:binary:logistic")),
    "rf-gpu":       (True, XgbBenchmark, metrics,
                     dict(rf_common_params, tree_method="gpu_exact",
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

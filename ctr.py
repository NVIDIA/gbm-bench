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
    X = df[df.columns.difference(["0"])]
    y = df["0"]
    return X,y

def prepareImplCommon(dbFolder, testSize, shuffle, datasetFileName, numRows):
    cols = [
      "0","1","2","3","4","5","6","7","8","9",
      "10","11","12","13","14","15","16","17","18","19",
      "20","21","22","23","24","25","26","27","28","29",
      "30","31","32","33","34","35","36","37","38","39"
    ]

    cat_cols = [
      "14","15","16","17","18","19",
      "20","21","22","23","24","25","26","27","28","29",
      "30","31","32","33","34","35","36","37","38","39"
    ]

    start = time.time()
    pklFile = os.path.join(dbFolder, "%s-%d.pkl" % (datasetFileName, numRows))
    if os.path.exists(pklFile):
        df = pd.read_pickle(pklFile)
    else:
        df1 =  pd.read_csv(os.path.join(dbFolder, datasetFileName), sep='\t', names=cols)
        df2 = convert_cols_categorical_to_numeric(df1, col_list=cat_cols)
        df = df2
        df.to_pickle(pklFile)
    X, y = generate_feables(df)
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=None,
                                                        shuffle=False,
                                                        random_state=42,
                                                        test_size=testSize)
    del X, y
    load_time = time.time() - start
    print("Criteo CTR dataset loaded in %.2fs" % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)

def prepareImpl(dbFolder, testSize, shuffle, nrows):
    rows = 2e7 if nrows is None else nrows
    return prepareImplCommon(dbFolder, testSize, shuffle, "day_1-1M.gz", rows)
    # return prepareImplCommon(dbFolder, testSize, shuffle, "day_1.npy", rows)

def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 0.01, True, nrows)


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 100

xgb_common_params = {
    "eta":              0.1,
    "gamma":            0.1,
    "learning_rate":    0.1,
    "max_depth":        8,
    "max_leaves":       2**8,
    "min_child_weight": 30,
    "num_round":        nTrees,
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
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
    "xgb-cpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="gpu:binary:logistic")),
    "xgb-gpu-hist": (True, XgbBenchmark, metrics,
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

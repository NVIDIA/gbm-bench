# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/06_HIGGS.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/06_HIGGS_GPU.ipynb
# source: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

from __future__ import print_function

import os
import time
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from new_utils import *


def generate_feables(df):
    X = df[df.columns.difference(["boson"])]
    y = df["boson"]
    return X, y

def prepareImpl(dbFolder, test_size, shuffle):
    # reading compressed csv is supported in pandas
    csv_name = "HIGGS.csv.gz"
    cols = [
        "boson", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude",
        "missing_energy_phi", "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
        "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag", "jet_3_pt", "jet_3_eta",
        "jet_3_phi", "jet_3_b-tag", "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
        "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
    ]
    csv_file = os.path.join(dbFolder, csv_name)
    start = time.time()
    df = pd.read_csv(csv_file, names=cols, dtype=np.float32)
    X, y = generate_feables(df)
    strati = y if shuffle else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=strati,
                                                        random_state=77,
                                                        test_size=test_size,
                                                        shuffle=shuffle)
    load_time = time.time() - start
    print("Higgs dataset loaded in %.2fs" % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)


def prepare(dbFolder):
    return prepareImpl(dbFolder, 500000, True)


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()

xgb_common_params = {
    "gamma":            0.1,
    "learning_rate":    0.1,
    "max_depth":        5,
    "max_leaves":       2**5,
    "min_child_weight": 1,
    "num_round":        200,
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "min_child_weight": 1,
    "min_split_gain":   0.1,
    "num_leaves":       2**5,
    "num_round":        200,
    "objective":        "binary",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            5,
    "iterations":       200,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
}

# NOTES: some tests are disabled!
#  . xgb-cpu takes almost an hour to train
#  . xgb-gpu runs out of memory
#  . lgb-gpu throws the following error
#   [LightGBM] [Fatal] Bug in GPU histogram! split 2237769: 5377646, smaller_leaf: 2237689, larger_leaf: 5377726
#  . cat-gpu throws the following error
#     what() -> "catboost/cuda/cuda_lib/devices_provider.h:96: CUDA error: all CUDA-capable devices are busy or unavailable 46"
benchmarks = {
    "xgb-cpu":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="binary:logistic")),
    "xgb-gpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="binary:logistic")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (False, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, catMetrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (False, CatBenchmark, catMetrics,
                     dict(cat_common_params, device_type="GPU")),
}

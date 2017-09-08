# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/06_HIGGS.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/06_HIGGS_GPU.ipynb
# source: https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz

from __future__ import print_function

import os
import time
import sys

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from utils import *


def generate_feables(df):
    X = df[df.columns.difference(['boson'])]
    y = df['boson']
    return X, y


def prepare(db_folder):
    # reading compressed csv is supported in pandas
    csv_name = 'HIGGS.csv.gz'
    cols = [
        'boson', 'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude',
        'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag', 'jet_3_pt', 'jet_3_eta',
        'jet_3_phi', 'jet_3_b-tag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
    csv_file = os.path.join(db_folder, csv_name)
    start = time.time()
    df = pd.read_csv(csv_file, names=cols, dtype=np.float32)
    X, y = generate_feables(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=77,
                                                        test_size=500000)
    load_time = time.time() - start
    print('Higgs dataset loaded in %.2fs' % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)

higgs_num_rounds = 200


class XgbGpuHiggs(XgbGpuBinaryBenchmark):
    num_rounds = higgs_num_rounds


class LgbmGpuHiggs(LgbmGpuBinaryBenchmark):
    num_rounds = higgs_num_rounds


xgb_cpu_model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=2,
    n_estimators=higgs_num_rounds,
    gamma=0.1,
    min_child_weight=1,
    reg_lambda=1,
    subsample=1,
    nthread=get_number_processors())

xgb_cpu_hist_model = xgb.XGBClassifier(
    max_depth=0,
    learning_rate=0.1,
    scale_pos_weight=2,
    n_estimators=higgs_num_rounds,
    gamma=0.1,
    min_child_weight=1,
    reg_lambda=1,
    subsample=1,
    max_leaves=2**5,
    grow_policy='lossguide',
    tree_method='hist',
    nthread=get_number_processors())

lgbm_cpu_model = lgbm.LGBMClassifier(
    num_leaves=2**5,
    learning_rate=0.1,
    scale_pos_weight=2,
    n_estimators=higgs_num_rounds,
    min_split_gain=0.1,
    min_child_weight=1,
    reg_lambda=1,
    subsample=1,
    nthread=get_number_processors())

# this actually runs out of memory, and therefore not used
xgb_gpu_params = {
    'max_depth': 2,
    'objective': 'binary:logistic',
    'min_child_weight': 1,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'gamma': 0.1,
    'reg_lamda': 1,
    'subsample': 1,
    'tree_method': 'exact',
    'updater': 'grow_gpu',
}

xgb_gpu_hist_params = {
    'max_depth': 6,
    'max_leaves': 2**5,
    'objective': 'binary:logistic',
    'min_child_weight': 1,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'gamma': 0.1,
    'reg_lamda': 1,
    'subsample': 1,
    'tree_method': 'gpu_hist',
#    'grow_policy': 'lossguide',
}

lgbm_gpu_params = {
    'num_leaves': 2**5,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'min_split_gain': 0.1,
    'min_child_weight': 1,
    'reg_lambda': 1,
    'subsample': 1,
    'objective': 'binary',
    'device': 'gpu',
    'task': 'train'
}

benchmarks = {
    # xgb-cpu takes almost an hour to train, and is therefore commented out by default
    # 'xgb-cpu': (CpuBinaryBenchmark, xgb_cpu_model),
    'xgb-cpu-hist': (CpuBinaryBenchmark, xgb_cpu_hist_model),
    'lgbm-cpu': (CpuBinaryBenchmark, lgbm_cpu_model),
    # xgb-gpu runs out of memory
    # 'xgb-gpu': (XgbGpuHiggs, xgb_gpu_params),
    'xgb-gpu-hist': (XgbGpuHiggs, xgb_gpu_hist_params),
    'lgbm-gpu': (LgbmGpuHiggs, lgbm_gpu_params)
}

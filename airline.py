# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/01_airline.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/01_airline_GPU.ipynb
# source: http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2

from __future__ import print_function

import os
import sys

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from conversion import *
from metrics import *
from utils import *


def generate_feables(df):
    X = df[df.columns.difference(['ArrDelay', 'ArrDelayBinary'])]
    y = df['ArrDelayBinary']
    return X,y


def prepare(db_folder):
    cols = [
        'Year', 'Month', 'DayofMonth', 'DayofWeek', 'CRSDepTime',
        'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'ActualElapsedTime',
        'Origin', 'Dest', 'Distance', 'Diverted', 'ArrDelay'
    ]

    # load the data as int16, and load only 2e7 rows (1/6 of the dataset) to avoid swapping
    dtype = np.int16
    dtype_columns = {
        'Year': dtype, 'Month': dtype, 'DayofMonth': dtype, 'DayofWeek': dtype,
        'CRSDepTime': dtype, 'CRSArrTime': dtype, 'FlightNum': dtype,
        'ActualElapsedTime': dtype, 'Distance': dtype, 'Diverted': dtype, 'ArrDelay': dtype,
    }
    start = time.time()
    df1 =  pd.read_csv(os.path.join(db_folder, 'airline_14col.data.bz2'),
                       names=cols, dtype=dtype_columns, nrows=2e7)
    df2 = convert_related_cols_categorical_to_numeric(df1, col_list=['Origin','Dest'])
    del df1
    df3 = convert_cols_categorical_to_numeric(df2, col_list='UniqueCarrier')
    del df2
    df = df3
    df['ArrDelayBinary'] = 1*(df['ArrDelay'] > 0)
    
    X, y = generate_feables(df)
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    del X, y
    load_time = time.time() - start
    print('Airline dataset loaded in %.2fs' % load_time, file=sys.stderr)
    
    return Data(X_train, X_test, y_train, y_test)


airline_num_rounds = 200

class CpuAirline(BinaryProbMixin, CpuBenchmark):

    def test(self):
        self.y_prob = np.clip(self.model.predict(self.X_test), 0.0001, 0.9999)


class XgbGpuAirline(XgbGpuBinaryBenchmark):

    num_rounds = airline_num_rounds


class LgbmGpuAirline(LgbmGpuBinaryBenchmark):

    num_rounds = airline_num_rounds


xgb_cpu_model = xgb.XGBRegressor(
    max_depth=8,
    n_estimators=airline_num_rounds,
    min_child_weight=30,
    learning_rate=0.1,
    scale_pos_weight=2,
    gamma=0.1,
    reg_lambda=1,
    subsample=1,
    n_jobs=get_number_processors(),
    random_state=77)

xgb_cpu_hist_model = xgb.XGBRegressor(
    max_depth=0,
    max_leaves=255,
    n_estimators=airline_num_rounds,
    min_child_weight=30,
    learning_rate=0.1,
    scale_pos_weight=2,
    gamma=0.1,
    reg_lambda=1,
    subsample=1,
    grow_policy='lossguide',
    tree_method='hist',
    n_jobs=get_number_processors(),
    random_state=77)

lgbm_cpu_model = lgbm.LGBMRegressor(
    num_leaves=255,
    n_estimators=airline_num_rounds,
    min_child_weight=30,
    learning_rate=0.1,
    scale_pos_weight=2,
    min_split_gain=0.1,
    reg_lambda=1,
    subsample=1,
    nthread=get_number_processors(),
    seed=77)

xgb_gpu_params = {
    'max_depth':8, #'max_depth':2,
    'objective':'binary:logistic',
    'min_child_weight':30,
    'eta':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'tree_method':'exact',
    'updater':'grow_gpu'
}

xgb_gpu_hist_params = {
    # the maximum depth supported by XGBoost is 15 (hard limit);
    # LGBM has significantly better accuracy,
    # presumably because it uses deeper trees
    'max_depth':12,
    'max_leaves':2**8,
    'objective':'binary:logistic',
    'min_child_weight':30,
    'eta':0.1,
    'scale_pos_weight':2,
    'gamma':0.1,
    'reg_lamda':1,
    'subsample':1,
    'tree_method':'gpu_hist',
}

lgbm_gpu_params = {
    'num_leaves': 2**8,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'min_split_gain': 0.1,
    'min_child_weight': 30,
    'reg_lambda': 1,
    'subsample': 1,
    'objective':'binary',
    'device': 'gpu',
    'task': 'train',
}

benchmarks = {

    # not benchmarked, as it takes very long to train
    # 'xgb-cpu': (CpuAirline, xgb_cpu_model),
    'xgb-cpu-hist': (CpuAirline, xgb_cpu_hist_model),
    'lgbm-cpu': (CpuAirline, lgbm_cpu_model),
    # GPU model runs out of memory
    # 'xgb-gpu': (XgbGpuAirline, xgb_gpu_params),
    'xgb-gpu-hist': (XgbGpuAirline, xgb_gpu_hist_params),
    'lgbm-gpu': (LgbmGpuAirline, lgbm_gpu_params),
}

# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/utils.py
from __future__ import print_function

import os
import multiprocessing
import subprocess
import sys
import time

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from metrics import *


# just a container for (X|y)_(train,test)
class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class Benchmark:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.model = None
        self.y_pred = None

    def run(self):
        results = {}
        self.prepare()
        # training
        start = time.time()
        self.train()
        train_time = time.time() - start
        # testing
        start = time.time()
        self.test()
        test_time = time.time() - start
        self.cleanup()
        return (train_time, test_time)

    def prepare(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def cleanup(self):
        if self.model is not None:
            del self.model


class XgbBenchmark(Benchmark):
    def prepare(self):
        self.dtrain = xgb.DMatrix(data=self.data.X_train, label=self.data.y_train)
        self.dtest = xgb.DMatrix(data=self.data.X_test, label=self.data.y_test)

    def train(self):
        self.model = xgb.train(self.params, self.dtrain)

    def test(self):
        self.y_pred = self.model.predict(self.dtest)

    def cleanup(self):
        del self.dtrain
        del self.dtest
        Benchmark.cleanup(self)


class LgbBenchmark(Benchmark):
    def prepare(self):
        self.dtrain = lgb.Dataset(self.data.X_train, self.data.y_train,
                                  free_raw_data=False)

    def train(self):
        self.model = lgb.train(self.params, self.dtrain)

    def test(self):
        self.y_pred = self.model.predict(self.data.X_test)

    def cleanup(self):
        self.model.free_dataset()
        del self.dtrain
        Benchmark.cleanup(self)


class CatBenchmark(Benchmark):
    def prepare(self):
        # NOTE: HACK!!
        # Due to some issue with CatBoostClassifier class we need to explicitly
        # set the below params to None, or else we get exceptions!
        params = self.params
        params['store_all_simple_ctr'] = None
        params['rsm'] = None
        self.model = cat.CatBoostClassifier(**params)

    def train(self):
        self.model.fit(self.data.X_train, self.data.y_train)

    def test(self):
        self.y_pred = self.model.predict_proba(self.data.X_test)


# set this to override the return value of get_number_processors()
number_processors_override = None

def get_number_processors():
    if number_processors_override is not None:
        return number_processors_override
    try:
        num = os.cpu_count()
    except:
        num = multiprocessing.cpu_count()
    return num  


def print_sys_info():
    print("System  : %s" % sys.version)
    print("Xgboost : %s" % os.getenv("XG_COMMIT_ID"))
    print("LightGBM: %s" % os.getenv("LG_COMMIT_ID"))
    print("CatBoost: %s" % os.getenv("CAT_COMMIT_ID"))
    print("#jobs   : %d" % get_number_processors())


def unarchive(db_folder, unarchiver, archive, target_file):
    # do not unzip if the target is there
    target_path = os.path.join(db_folder, target_file)
    if not os.path.exists(target_path):
        print('Unzipping the data...')
        subprocess.check_call('cd %s && %s %s' % \
                              (db_folder, unarchiver, archive), shell=True)
    else:
        print('Skipping data unzip')


def unzip(db_folder, archive, target_file):
    unarchive(db_folder, 'unzip', archive, target_file)

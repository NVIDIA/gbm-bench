# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/utils.py
from __future__ import print_function

import os
import multiprocessing
import subprocess
import sys
import time

import lightgbm as lgb
import xgboost as xgb

from metrics import *


# just a container for (X|y)_(train,test)
class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class Benchmark:
    # params is either model or parameters for the model
    def __init__(self, data, params):
        # set the data
        self.X_train = data.X_train
        self.X_test = data.X_test
        self.y_train = data.y_train
        self.y_test = data.y_test
        self.params = params

    def run(self):
        results = {}
        # additional data and model preparation, if necessary
        self.prepare()
        # training time
        start = time.time()
        self.train()
        results['train_time'] = time.time() - start
        # testing time
        start = time.time()
        self.test()
        results['test_time'] = time.time() - start
        # accuracy
        results['accuracy'] = self.accuracy()
        # cleanup, if necessary
        self.cleanup()
        return results

    def prepare(self):
        pass

    def cleanup(self):
        pass

    
class CpuBenchmark(Benchmark):
    def __init__(self, data, model):
        Benchmark.__init__(self, data, model)
        self.model = model

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        self.y_pred = self.model.predict(self.X_test)


# CPU benchmark with binary classification metrics
class CpuBinaryBenchmark(CpuBenchmark):
    def accuracy(self):
        return classification_metrics(self.y_test, self.y_pred)   


class GpuBenchmark(Benchmark):
    # number of boosting rounds
    num_rounds = None

    def cleanup(self):
        del self.model


class XgbGpuBenchmark(GpuBenchmark):
    def prepare(self):
        self.dtrain = xgb.DMatrix(data=self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(data=self.X_test, label=self.y_test)

    def train(self):
        self.model = xgb.train(self.params, self.dtrain, num_boost_round=self.num_rounds)

    def test(self):
        self.y_prob = self.model.predict(self.dtest)


class LgbmGpuBenchmark(GpuBenchmark):
    def prepare(self):
        self.dtrain = lgb.Dataset(self.X_train, self.y_train, free_raw_data=False)

    def train(self):
        self.model = lgb.train(self.params, self.dtrain, num_boost_round=self.num_rounds)

    def test(self):
        self.y_prob = self.model.predict(self.X_test)

    def cleanup(self):
        self.model.free_dataset()
        del self.dtrain
        GpuBenchmark.cleanup(self)


# mixin for binary accuracy computation for predictions expressed as probability
class BinaryProbMixin:
    def accuracy(self):
        y_pred = binarize_prediction(self.y_prob)
        results = classification_metrics_binary(self.y_test, y_pred)
        results2 = classification_metrics_binary_prob(self.y_test, self.y_prob)
        results.update(results2)
        return results

    
class XgbGpuBinaryBenchmark(BinaryProbMixin, XgbGpuBenchmark):
    pass


class LgbmGpuBinaryBenchmark(BinaryProbMixin, LgbmGpuBenchmark):
    pass

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

# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/utils.py
from __future__ import print_function

import os
import multiprocessing
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


def get_number_processors():
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

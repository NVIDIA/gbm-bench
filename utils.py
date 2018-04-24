# MIT License
#
# Copyright (c) Microsoft Corporation. All rights reserved.
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

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

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
        self.model = xgb.train(self.params, self.dtrain, self.params['num_round'])

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
        # CB_THREAD_LIMIT is set to 56 in catboost source!
        if 'thread_count' in params and params['thread_count'] > 56:
            print("Warning! catboost sets max-thread-count to 56!")
            params['thread_count'] = 56
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

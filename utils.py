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

import multiprocessing
import os
import subprocess
import sys
import time

import catboost as cat
import dask as dask
import dask.dataframe as ddf
import dask.distributed as dd
import dask_gdf as dgdf
import dask_xgboost as dxgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import cudf.dataframe as gdf
import xgboost as xgb

from metrics import *

# convert an object to GDF
def to_gdf(obj):
    if isinstance(obj, pd.DataFrame):
        return gdf.DataFrame.from_pandas(obj)
    elif isinstance(obj, pd.Series):
        return gdf.DataFrame.from_pandas(obj.to_frame())
    elif isinstance(obj, gdf.DataFrame):
        return obj
    elif isinstance(obj, gdf.Series):
        return obj
    raise ValueError('type %s not supported' % type(obj))


# just a container for (X|y)_(train,test)
class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def to_gdf(self):
        X_train_gdf = to_gdf(self.X_train)
        X_test_gdf = to_gdf(self.X_test)
        y_train_gdf = to_gdf(self.y_train)
        y_test_gdf = to_gdf(self.y_test)
        return Data(X_train_gdf, X_test_gdf, y_train_gdf, y_test_gdf)

    def to_dask(self, nworkers):
        X_train_dask = ddf.from_pandas(self.X_train, npartitions=nworkers)
        X_test_dask = ddf.from_pandas(self.X_test, npartitions=nworkers)
        # y_train_dask = ddf.from_pandas(self.y_train, npartitions=nworkers)
        # y_test_dask = ddf.from_pandas(self.y_test, npartitions=nworkers)
        y_train_dask = ddf.from_pandas(self.y_train.to_frame(), npartitions=nworkers)
        y_test_dask = ddf.from_pandas(self.y_test.to_frame(), npartitions=nworkers)
        X_train_dask, X_test_dask, y_train_dask, y_test_dask = dask.persist(
            X_train_dask, X_test_dask, y_train_dask, y_test_dask)
        return Data(X_train_dask, X_test_dask, y_train_dask, y_test_dask)

    def to_dask_gdf(self, nworkers):
        d1 = self.to_dask(nworkers)
        X_train_dgdf = dgdf.from_dask_dataframe(d1.X_train)
        #print(X_train_dgdf.columns)
        X_test_dgdf = dgdf.from_dask_dataframe(d1.X_test)
        #print(X_test_dgdf.columns)
        y_train_dgdf = dgdf.from_dask_dataframe(d1.y_train)
        y_test_dgdf = dgdf.from_dask_dataframe(d1.y_test)
        X_train_dgdf, X_test_dgdf, y_train_dgdf, y_test_dgdf = dask.persist(
            X_train_dgdf, X_test_dgdf, y_train_dgdf, y_test_dgdf)
        return Data(X_train_dgdf, X_test_dgdf, y_train_dgdf, y_test_dgdf)

    def dask_wait(self):
        dd.wait(self.X_train)
        dd.wait(self.X_test)
        dd.wait(self.y_train)
        dd.wait(self.y_test)

    def y_test_matrix(self):
        y = self.y_test
        if isinstance(y, gdf.DataFrame):
            return y.as_matrix()
        elif isinstance(y, gdf.Series):
            return y.to_array()
        elif isinstance(y, (ddf.DataFrame, ddf.Series)):
            return y.persist().compute()
        elif isinstance(y, (dgdf.DataFrame, dgdf.Series)):
            return y.to_dask_dataframe().persist().compute()
        return y


class Benchmark:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.model = None
        self.y_pred = None

    def __enter__(self):
        pass

    def df_prepare(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None:
            del self.model

    def run(self):
        # preparing the df: converting them to GDF if necessary; this is not timed
        self.df_prepare()
        
        # preparing; calling DMatrix ctors for train and test
        start = time.time()
        self.prepare()
        prepare_time = time.time() - start

        # training
        start = time.time()
        self.train()
        train_time = time.time() - start

        # testing
        start = time.time()
        self.test()
        test_time = time.time() - start
        return (prepare_time, train_time, test_time)

    def prepare(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def y_test_matrix(self):
        return self.data.y_test_matrix()


# dask environment; manages the processes
# TODO: actually make it customizable
class DaskEnv:
    scheduler = None
    workers = None
    params = None
    nworkers = None
    
    def __init__(self, params):
        self.params = params
        self.nworkers = params['nworkers']

    def start(self):
        # output redirect
        output = subprocess.DEVNULL
        if 'verbose' in self.params and self.params['verbose'] > 0:
            # setting output to None inherits the descriptors
            # of the parent process
            output = None
        
        # start the scheduler
        self.scheduler = subprocess.Popen(
            ['dask-scheduler', '--port=8786', '--host=127.0.0.1'],
            stdout=output, stderr=output)
        time.sleep(1)
        # start the workers with the right devices
        ram_fraction = 1.0 / self.nworkers
        self.workers = []
        for i in range(self.nworkers):
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': '%d' % i})
            self.workers.append(subprocess.Popen(
                ['dask-worker',
                 '127.0.0.1:8786',
                 '--no-nanny',
                 '--memory-limit=%.3f' % ram_fraction,
                 '--nprocs=1', '--nthreads=1'],
                env=env, stdout=output, stderr=output))
        time.sleep(2)

    def stop(self):
        time.sleep(2)
        for i in range(self.nworkers):
            self.workers[i].kill()
        self.scheduler.kill()


# runs the benchmark using dask-xgboost
class XgbDaskBenchmark(Benchmark):
    dask_env = None
    ip_port = '127.0.0.1:8786'  # ip:port for the Client
    client = None

    def __init__(self, data, params):
        Benchmark.__init__(self, data, params)
        # 'distributed_dask' must be True
        self.params['distributed_dask']  = True
        # interpret GPUs as the number of workers
        self.params['dask_nworkers'] = self.params['n_gpus']
        self.params['n_gpus'] = 1

    def __enter__(self):
        Benchmark.__enter__(self)
        # set up the dask environment
        self.dask_env = DaskEnv({
            'nworkers': self.params['dask_nworkers'],
            'verbose': self.params['debug_verbose'] if 'debug_verbose' in self.params else 0
        })
        self.dask_env.start()
        self.client = dd.Client(self.ip_port)

    def df_prepare(self):
        # prepare the dask dataframes
        self.data = self.data.to_dask(self.dask_env.nworkers)
        self.data.dask_wait()

    def train(self):
        bst = dxgb.train(self.client, self.params, self.data.X_train,
                         self.data.y_train,
                         num_boost_round=self.params['num_round'])
        if isinstance(bst, list):
            # comment out this loop for higher performance
            for i in range(len(bst)):
                bst[i].dump_model('file-%d.model' % i)
            bst = bst[0]
        self.model = bst

    def test(self):
        self.y_pred = dxgb.predict(
            self.client, self.model, self.data.X_test).persist().compute()
        dd.wait(self.y_pred)

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        del self.client
        self.dask_env.stop()
        Benchmark.__exit__(self, exc_type, exc_value, traceback)


class XgbDaskGdfBenchmark(XgbDaskBenchmark):

    def __init__(self, data, params):
        XgbDaskBenchmark.__init__(self, data, params)

    def df_prepare(self):
        # prepare the dask-gdf dataframes
        self.data = self.data.to_dask_gdf(self.dask_env.nworkers)
        self.data.dask_wait()

    def test(self):
        self.y_pred = dxgb.predict(
            self.client, self.model, self.data.X_test).persist()
        # merge the predictions in host memory, in case the dataset is large
        self.y_pred = self.y_pred.to_dask_dataframe().compute()
        dd.wait(self.y_pred)


class XgbBenchmark(Benchmark):
    def prepare(self):
        self.dtrain = xgb.DMatrix(data=self.data.X_train, label=self.data.y_train)
        self.dtest = xgb.DMatrix(data=self.data.X_test, label=self.data.y_test)

    def train(self):
        self.model = xgb.train(self.params, self.dtrain, self.params['num_round'])

    def test(self):
        self.y_pred = self.model.predict(self.dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.dtrain
        del self.dtest
        Benchmark.__exit__(self, exc_type, exc_value, traceback)


class XgbGdfBenchmark(XgbBenchmark):
    def df_prepare(self):
        XgbBenchmark.df_prepare(self)
        self.data = self.data.to_gdf()


class LgbBenchmark(Benchmark):
    def prepare(self):
        self.dtrain = lgb.Dataset(self.data.X_train, self.data.y_train,
                                  free_raw_data=False)

    def train(self):
        self.model = lgb.train(self.params, self.dtrain)

    def test(self):
        self.y_pred = self.model.predict(self.data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.free_dataset()
        del self.dtrain
        Benchmark.__exit__(self, exc_type, exc_value, traceback)


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

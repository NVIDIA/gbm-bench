# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from abc import ABC, abstractmethod
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from datasets import LearningTask
import dask.dataframe as dd
import time
import GPUtil
from dask.distributed import Client, LocalCluster


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


class Algorithm(ABC):
    @staticmethod
    def create(name):
        if name == 'xgb-gpu':
            return XgbGPUHistAlgorithm()
        if name == 'xgb-gpu-dask':
            return XgbGPUHistDaskAlgorithm()
        if name == 'xgb-cpu':
            return XgbCPUHistAlgorithm()
        if name == 'lgbm-cpu':
            return LgbmCPUAlgorithm()
        if name == 'lgbm-gpu':
            return LgbmGPUAlgorithm()
        if name == 'cat-cpu':
            return CatCPUAlgorithm()
        if name == 'cat-gpu':
            return CatGPUAlgorithm()
        raise ValueError("Unknown algorithm: " + name)

    @abstractmethod
    def fit(self, data, args):
        pass

    @abstractmethod
    def test(self, data):
        pass

    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass


# learning parameters shared by all algorithms, using the xgboost convention
shared_params = {"max_depth": 8, "learning_rate": 0.1,
                 "reg_lambda": 1}


class XgbAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({"max_leaves": 256,
                       "nthread": args.cpus})
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "reg:squarederror"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary:logistic"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multi:softmax"
            params["num_class"] = np.max(data.y_test) + 1
        params.update(args.extra)
        return params

    def fit(self, data, args):
        self.dtrain = xgb.DMatrix(data.X_train, data.y_train)
        self.dtest = xgb.DMatrix(data.X_test, data.y_test)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = xgb.train(params, self.dtrain, args.ntrees)
        return t.interval

    def test(self, data):
        return self.model.predict(self.dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        del self.dtrain
        del self.dtest


class XgbGPUHistAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbGPUHistAlgorithm, self).configure(data, args)
        params.update({"tree_method": "gpu_hist", "n_gpus": args.gpus})
        return params


class XgbGPUHistDaskAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbGPUHistDaskAlgorithm, self).configure(data, args)
        params.update({"tree_method": "gpu_hist"})
        return params

    def train(self, X, y, params, devices, args):
        params['gpu_id'] = devices[xgb.rabit.get_rank()]
        dtrain = xgb.dask.create_worker_dmatrix(X, y)
        with Timer() as t:
            model = xgb.train(params, dtrain, args.ntrees)
        return model, t.interval

    def fit(self, data, args):
        params = self.configure(data, args)
        devices = GPUtil.getAvailable(limit=32 if args.gpus < 0 else args.gpus)
        cluster = LocalCluster(n_workers=len(devices), threads_per_worker=args.cpus // len(devices),
                              local_dir="/opt/gbm-datasets")
        client = Client(cluster)
        partition_size = 100000
        try:
            X = dd.from_array(data.X_train, partition_size)
            y = dd.from_array(data.y_train, partition_size)
        except:
            X = dd.from_pandas(data.X_train, partition_size)
            y = dd.from_pandas(data.y_train, partition_size)
        result = xgb.dask.run(client, self.train, X, y, params, devices, args)
        self.model, time = next(iter(result.values()))
        client.close()
        return time

    def test(self, data):
        dtest = xgb.DMatrix(data.X_test, data.y_test)
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class XgbCPUHistAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbCPUHistAlgorithm, self).configure(data, args)
        params.update({"tree_method": "hist"})
        return params


class LgbmAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({"max_leaves": 256,
                       "nthread": args.cpus})
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "regression"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "multiclass"
            params["num_class"] = np.max(data.y_test) + 1
        params.update(args.extra)
        return params

    def fit(self, data, args):
        self.dtrain = lgb.Dataset(data.X_train, data.y_train,
                                  free_raw_data=False)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = lgb.train(params, self.dtrain, args.ntrees)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            prob = self.model.predict(data.X_test)
            return np.argmax(prob, axis=1)
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.free_dataset()
        del self.model
        del self.dtrain


class LgbmCPUAlgorithm(LgbmAlgorithm):
    pass


class LgbmGPUAlgorithm(LgbmAlgorithm):
    def configure(self, data, args):
        params = super(LgbmGPUAlgorithm, self).configure(data, args)
        params.update({"device": "gpu"})
        return params


class CatAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        params.update({
            "thread_count": args.cpus})
        if args.gpus >= 0:
            params["devices"] = "0-" + str(args.gpus)

        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "RMSE"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "Logloss"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        elif data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            params["objective"] = "MultiClassOneVsAll"
            params["classes_count"] = np.max(data.y_test) + 1
        params.update(args.extra)
        return params

    def fit(self, data, args):
        self.dtrain = cat.Pool(data.X_train, data.y_train)
        params = self.configure(data, args)
        params["iterations"] = args.ntrees
        self.model = cat.CatBoost(params)
        with Timer() as t:
            self.model.fit(self.dtrain)
        return t.interval

    def test(self, data):
        dtest = cat.Pool(data.X_test)
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            prob = self.model.predict(dtest)
            return np.argmax(prob, axis=1)
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        del self.dtrain


class CatCPUAlgorithm(CatAlgorithm):
    def configure(self, data, args):
        params = super(CatCPUAlgorithm, self).configure(data, args)
        params.update({"task_type": "CPU"})
        return params


class CatGPUAlgorithm(CatAlgorithm):
    def configure(self, data, args):
        params = super(CatGPUAlgorithm, self).configure(data, args)
        params.update({"task_type": "GPU"})
        return params

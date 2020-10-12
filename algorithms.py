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
import time
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import xgboost as xgb
import cudf

try:
    import catboost as cat
except ImportError:
    cat = None
try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None
try:
    import dask_xgboost as dxgb
except ImportError:
    dxgb = None
try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier as skhgb
except ImportError:
    skhgb = None
try:
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor as skhgb_r
except ImportError:
    skhgb_r = None
try: 
    from sklearn.ensemble import GradientBoostingClassifier as skgb
except ImportError:
    skgb = None
try: 
    from sklearn.ensemble import GradientBoostingRegressor as skgb_r
except ImportError:
    skgb_r = None
try:
    from sklearn.ensemble import RandomForestClassifier as skrf
except ImportError:
    skrf = None
try:
    from sklearn.ensemble import RandomForestRegressor as skrf_r
except ImportError:
    skrf_r = None
try:
    from cuml.ensemble import RandomForestClassifier as cumlrf
except ImportError:
    cumlrf = None
try:
    from cuml.ensemble import RandomForestRegressor as cumlrf_r
except ImportError:
    cumlrf_r = None

from datasets import LearningTask


class Timer:
    def __init__(self):
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class Algorithm(ABC):
    @staticmethod
    def create(name):  # pylint: disable=too-many-return-statements
        if name == 'xgb-gpu':
            return XgbGPUHistAlgorithm()
        if name == 'xgb-gpu-dask':
            return XgbGPUHistDaskAlgorithm()
        if name == 'xgb-gpu-dask-old':
            return XgbGPUHistDaskOldAlgorithm()
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
        if name == 'skhgb':
            return SkHistAlgorithm()
        if name == 'skgb':
            return SkGradientAlgorithm()
        if name == 'skrf':
            return SkRandomForestAlgorithm()
        if name == 'cumlrf':
            return CumlRfAlgorithm()
        raise ValueError("Unknown algorithm: " + name)

    def __init__(self):
        self.model = None

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

class CumlRfAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        del params["reg_lambda"]
        del params["learning_rate"]
        params["n_estimators"] = args.ntrees
        params.update(args.extra)
        return params

    def fit(self, data, args):
        params = self.configure(data, args)
        if data.learning_task == LearningTask.REGRESSION:
            with Timer() as t:
                self.model = cumlrf_r(**params).fit(data.X_train, data.y_train)
            return t.interval
        else:
            with Timer() as t:
                self.model = cumlrf(**params).fit(data.X_train, data.y_train)
            return t.interval

    def test(self, data):
        return self.model.predict(data.X_test)
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        
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
        dtrain = xgb.DMatrix(data.X_train, data.y_train)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = xgb.train(params, dtrain, args.ntrees)
        return t.interval

    def test(self, data):
        dtest = xgb.DMatrix(data.X_test, data.y_test)
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class XgbGPUHistAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbGPUHistAlgorithm, self).configure(data, args)
        params.update({"tree_method": "gpu_hist", "gpu_id": 0})
        return params
        
class SkRandomForestAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        del params["reg_lambda"]
        del params["learning_rate"]
        params["n_estimators"] = args.ntrees
        params.update(args.extra)
        return params

    def fit(self, data, args):
        params = self.configure(data, args)
        if data.learning_task == LearningTask.REGRESSION:
            with Timer() as t:
                self.model = skrf_r(**params).fit(data.X_train, data.y_train)
            return t.interval
        else:
            with Timer() as t:
                self.model = skrf(**params).fit(data.X_train, data.y_train)
            return t.interval

    def test(self, data):
        return self.model.predict(data.X_test)
    
    def __exit__(self, exc_type, exc_value, traceback):
        del self.model

class SkGradientAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        del params["reg_lambda"]
        del params["learning_rate"]
        params["n_estimators"] = args.ntrees
        params.update(args.extra)
        return params
    
    def fit(self, data, args):
        params = self.configure(data, args)
        if data.learning_task == LearningTask.REGRESSION:
            with Timer() as t:
                self.model = skgb_r(**params).fit(data.X_train, data.y_train)
            return t.interval
        else:
            with Timer() as t:
                self.model = skgb(**params).fit(data.X_train, data.y_train)
            return t.interval
    
    def test(self, data):
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model

class SkHistAlgorithm(Algorithm):
    def configure(self, data, args):
        params = shared_params.copy()
        del params["reg_lambda"]
        del params["learning_rate"]
        params["n_estimators"] = args.ntrees
        params.update(args.extra)
        return params
    
    def fit(self, data, args):
        params = self.configure(data, args)
        if data.learning_task == LearningTask.REGRESSION:
            with Timer() as t:
                self.model = skhgb_r(**params).fit(data.X_train, data.y_train)
            return t.interval
        else:
            with Timer() as t:
                self.model = skhgb(**params).fit(data.X_train, data.y_train)
            return t.interval

    def test(self, data):
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
    

class XgbGPUHistDaskAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbGPUHistDaskAlgorithm, self).configure(data, args)
        params.update({"tree_method": "gpu_hist"})
        del params['nthread']  # This is handled by dask
        return params

    def get_slices(self, n_slices, X, y):
        n_rows_worker = int(np.ceil(len(y) / n_slices))
        indices = []
        count = 0
        for _ in range(0, n_slices - 1):
            indices.append(min(count + n_rows_worker, len(y)))
            count += n_rows_worker
        return np.split(X, indices), np.split(y, indices)

    def fit(self, data, args):
        params = self.configure(data, args)
        n_workers = None if args.gpus < 0 else args.gpus
        cluster = LocalCUDACluster(n_workers=n_workers,
                                   local_directory=args.root)
        client = Client(cluster)
        n_partitions = len(client.scheduler_info()['workers'])
        X_sliced, y_sliced = self.get_slices(n_partitions,
                                             data.X_train, data.y_train)
        X = da.concatenate([da.from_array(sub_array) for sub_array in X_sliced])
        X = X.rechunk((X_sliced[0].shape[0], data.X_train.shape[1]))
        y = da.concatenate([da.from_array(sub_array) for sub_array in y_sliced])
        y = y.rechunk(X.chunksize[0])
        dtrain = xgb.dask.DaskDMatrix(client, X, y)
        with Timer() as t:
            output = xgb.dask.train(client, params, dtrain, num_boost_round=args.ntrees)
        self.model = output['booster']
        client.close()
        cluster.close()
        return t.interval

    def test(self, data):
        dtest = xgb.DMatrix(data.X_test, data.y_test)
        self.model.set_param({'predictor': 'gpu_predictor'})
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


class XgbGPUHistDaskOldAlgorithm(XgbAlgorithm):
    def configure(self, data, args):
        params = super(XgbGPUHistDaskOldAlgorithm, self).configure(data, args)
        params.update({"tree_method": "gpu_hist", "nthread": 1})
        return params

    def fit(self, data, args):
        params = self.configure(data, args)
        cluster = LocalCUDACluster(n_workers=None if args.gpus < 0 else args.gpus,
                                   local_directory=args.root)
        client = Client(cluster)
        partition_size = 1000
        if isinstance(data.X_train, np.ndarray):
            X = dd.from_array(data.X_train, partition_size)
            y = dd.from_array(data.y_train, partition_size)
        else:
            X = dd.from_pandas(data.X_train, partition_size)
            y = dd.from_pandas(data.y_train, partition_size)
        X.columns = [str(i) for i in range(0, X.shape[1])]
        with Timer() as t:
            self.model = dxgb.train(client, params, X, y, num_boost_round=args.ntrees)

        client.close()
        return t.interval

    def test(self, data):
        if isinstance(data.X_test, np.ndarray):
            data.X_test = pd.DataFrame(data=data.X_test, columns=np.arange(0,
                                                                           data.X_test.shape[1]),
                                       index=np.arange(0, data.X_test.shape[0]))
        data.X_test.columns = [str(i) for i in range(0, data.X_test.shape[1])]
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
        dtrain = lgb.Dataset(data.X_train, data.y_train,
                             free_raw_data=False)
        params = self.configure(data, args)
        with Timer() as t:
            self.model = lgb.train(params, dtrain, args.ntrees)
        return t.interval

    def test(self, data):
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            prob = self.model.predict(data.X_test)
            return np.argmax(prob, axis=1)
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.free_dataset()
        del self.model


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
        dtrain = cat.Pool(data.X_train, data.y_train)
        params = self.configure(data, args)
        params["iterations"] = args.ntrees
        self.model = cat.CatBoost(params)
        with Timer() as t:
            self.model.fit(dtrain)
        return t.interval

    def test(self, data):
        dtest = cat.Pool(data.X_test)
        if data.learning_task == LearningTask.MULTICLASS_CLASSIFICATION:
            prob = self.model.predict(dtest)
            return np.argmax(prob, axis=1)
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model


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

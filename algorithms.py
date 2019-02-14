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


class Algorithm(ABC):
    @staticmethod
    def create(name, data):
        if name == 'xgb-gpu':
            return XgbGPUHistAlgorithm(data)
        if name == 'xgb-cpu':
            return XgbCPUHistAlgorithm(data)
        if name == 'lgbm-cpu':
            return LgbmCPUAlgorithm(data)
        if name == 'lgbm-gpu':
            return LgbmGPUAlgorithm(data)
        if name == 'cat-cpu':
            return CatCPUAlgorithm(data)
        if name == 'cat-gpu':
            return CatGPUAlgorithm(data)
        raise ValueError("Unknown algorithm: " + name)

    @abstractmethod
    def fit(self, data, num_trees):
        pass

    @abstractmethod
    def test(self, data):
        pass

    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass


class XgbAlgorithm(Algorithm):
    def __init__(self, data):
        self.model = None
        self.dtrain = xgb.DMatrix(data.X_train, data.y_train)
        self.dtest = xgb.DMatrix(data.X_test, data.y_test)

    def configure(self, data):
        params = {"max_depth": 8, "max_leaves": 256, "learning_rate": 0.1, "min_child_weight": 1,
                  "reg_lambda": 1, "reg_alpha": 1}
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "reg:linear"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary:logistic"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        return params

    def fit(self, data, num_trees):
        params = self.configure(data)
        self.model = xgb.train(params, self.dtrain, num_trees)

    def test(self, data):
        return self.model.predict(self.dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        del self.dtrain
        del self.dtest


class XgbGPUHistAlgorithm(XgbAlgorithm):
    def configure(self, data):
        params = super(XgbGPUHistAlgorithm, self).configure(data)
        params.update({"tree_method": "gpu_hist"})
        return params


class XgbCPUHistAlgorithm(XgbAlgorithm):
    def configure(self, data):
        params = super(XgbCPUHistAlgorithm, self).configure(data)
        params.update({"tree_method": "hist"})
        return params


class LgbmAlgorithm(Algorithm):
    def __init__(self, data):
        self.dtrain = lgb.Dataset(data.X_train, data.y_train,
                                  free_raw_data=False)
        self.model = None

    def configure(self, data):
        params = {"max_depth": 8, "max_leaves": 256, "learning_rate": 0.1, "min_child_weight": 1,
                  "reg_lambda": 1, "reg_alpha": 1}
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "regression"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "binary"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        return params

    def fit(self, data, num_trees):
        params = self.configure(data)
        self.model = lgb.train(params, self.dtrain, num_trees)

    def test(self, data):
        return self.model.predict(data.X_test)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.free_dataset()
        del self.model
        del self.dtrain


class LgbmCPUAlgorithm(LgbmAlgorithm):
    pass


class LgbmGPUAlgorithm(LgbmAlgorithm):
    def configure(self, data):
        params = super(LgbmGPUAlgorithm, self).configure(data)
        params.update({"device": "gpu"})
        return params


class CatAlgorithm(Algorithm):
    def __init__(self, data):
        self.dtrain = cat.Pool(data.X_train, data.y_train)
        self.model = None

    def configure(self, data):
        params = {"max_depth": 8, "learning_rate": 0.1,
                  "reg_lambda": 1}
        if data.learning_task == LearningTask.REGRESSION:
            params["objective"] = "RMSE"
        elif data.learning_task == LearningTask.CLASSIFICATION:
            params["objective"] = "Logloss"
            params["scale_pos_weight"] = len(data.y_train) / np.count_nonzero(data.y_train)
        return params

    def fit(self, data, num_trees):
        params = self.configure(data)
        params["iterations"] = num_trees
        self.model = cat.CatBoost(params)
        self.model.fit(self.dtrain)

    def test(self, data):
        dtest = cat.Pool(data.X_test)
        return self.model.predict(dtest)

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        del self.dtrain


class CatCPUAlgorithm(CatAlgorithm):
    def configure(self, data):
        params = super(CatCPUAlgorithm, self).configure(data)
        params.update({"task_type": "CPU"})
        return params


class CatGPUAlgorithm(CatAlgorithm):
    def configure(self, data):
        params = super(CatGPUAlgorithm, self).configure(data)
        params.update({"task_type": "GPU"})
        return params

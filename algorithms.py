from abc import ABC, abstractmethod
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import numpy as np
from datasets import LearningTask


class Algorithm(ABC):
    def create(name, data):
        if name == 'xgb-gpu':
            return XgbGPUHistAlgorithm(data)
        elif name == 'xgb-cpu':
            return XgbCPUHistAlgorithm(data)
        elif name == 'lgbm-cpu':
            return LgbmCPUAlgorithm(data)
        elif name == 'lgbm-gpu':
            return LgbmGPUAlgorithm(data)
        elif name == 'cat-cpu':
            return CatCPUAlgorithm(data)
        elif name == 'cat-gpu':
            return CatGPUAlgorithm(data)
        else:
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
    def configure(self, data):
        return super(LgbmCPUAlgorithm, self).configure(data)


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


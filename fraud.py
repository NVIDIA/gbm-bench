# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection_GPU.ipynb
# source: https://www.kaggle.com/dalpozz/creditcardfraud

import json
import os
import pkg_resources
import subprocess
import sys
import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

import utils
from utils import *

def prepare(dbFolder):

    # unzip the data
    csv_file = os.path.join(dbFolder, 'creditcard.csv')
    if not os.path.exists(csv_file):
        print('Unzipping the data...')
        subprocess.check_call('cd %s && unzip creditcardfraud.zip' % dbFolder, shell=True)
    else:
        print('Skipping data unzip')
    df = pd.read_csv(csv_file)

    X = df[[col for col in df.columns if col.startswith('V')]].values
    y = df['Class'].values

    print('Features: ', X.shape)
    print('Labels: ', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)

    return Data(X_train, X_test, y_train, y_test)
    

class CpuFraudDetection(CpuBenchmark):

    def accuracy(self):
        return classification_metrics(self.y_test, self.y_pred)

    
class GpuFraudDetection:

    num_rounds = 100

    def accuracy(self):
        y_pred = binarize_prediction(self.y_prob)
        results = classification_metrics_binary(self.y_test, y_pred)
        results2 = classification_metrics_binary_prob(self.y_test, self.y_prob)
        results.update(results2)
        return results

    
class LgbmGpuFraudDetection(GpuFraudDetection, LgbmGpuBenchmark):
    pass


class XgbGpuFraudDetection(GpuFraudDetection, XgbGpuBenchmark):
    pass

    
def featurisers():
    pipeline_steps = [('scale', StandardScaler())]
    continuous_pipeline = Pipeline(steps=pipeline_steps)
    featurisers = [('continuous', continuous_pipeline)]
    return featurisers

xgb_cpu_model = Pipeline(
        steps=[('features', FeatureUnion(featurisers())),
               ('clf', XGBClassifier(
                    max_depth=3, 
                    learning_rate=0.1, 
                    scale_pos_weight=2,
                    n_estimators=100,
                    gamma=0.1,
                    min_child_weight=1,
                    reg_lambda=1,
                    subsample=1,
                    nthread=get_number_processors()))])

xgb_cpu_hist_model = Pipeline(
        steps=[('features', FeatureUnion(featurisers())),
               ('clf', 
                XGBClassifier(
                    max_depth=0, 
                    learning_rate=0.1, 
                    scale_pos_weight=2,
                    n_estimators=100,
                    gamma=0.1,
                    min_child_weight=1,
                    reg_lambda=1,
                    subsample=1,
                    max_leaves=2**3,
                    grow_policy='lossguide',
                    tree_method='hist',
                    nthread=get_number_processors()))])

lgbm_cpu_model = Pipeline(
        steps=[('features', FeatureUnion(featurisers())),
               ('clf', 
                LGBMClassifier(
                    num_leaves=2**3, 
                    learning_rate=0.1, 
                    scale_pos_weight=2,
                    n_estimators=100,
                    min_split_gain=0.1,
                    min_child_weight=1,
                    reg_lambda=1,
                    subsample=1,
                    nthread=get_number_processors()))])

xgb_gpu_params = {
    'max_depth':3, 
    'objective':'binary:logistic', 
    'min_child_weight':1, 
    'eta':0.1, 
    'colsample_bytree':1, 
    'scale_pos_weight':2, 
    'gamma':0.1, 
    'reg_lamda':1, 
    'subsample':1,
    'tree_method':'exact', 
    'updater':'grow_gpu',
}

xgb_gpu_hist_params = {
    'max_depth':0, 
    'objective':'binary:logistic', 
    'min_child_weight':1, 
    'eta':0.1, 
    'colsample_bytree':0.80, 
    'scale_pos_weight':2, 
    'gamma':0.1, 
    'reg_lamda':1, 
    'subsample':1,
    'tree_method':'hist', 
    'max_leaves':2**3, 
    'grow_policy':'lossguide',
}

lgbm_gpu_params = {
    'num_leaves': 2**3,
    'learning_rate': 0.1,
    'scale_pos_weight': 2,
    'min_split_gain': 0.1,
    'min_child_weight': 1,
    'reg_lambda': 1,
    'subsample': 1,
    'objective':'binary',
    'task': 'train',    
}

benchmarks = {
    'xgb-cpu':      (CpuFraudDetection, xgb_cpu_model),
    'xgb-cpu-hist': (CpuFraudDetection, xgb_cpu_hist_model),
    'lgbm-cpu':     (CpuFraudDetection, lgbm_cpu_model),
    'xgb-gpu': (XgbGpuFraudDetection, xgb_gpu_params),
    'xgb-gpu-hist': (XgbGpuFraudDetection, xgb_gpu_hist_params),
    'lgbm-gpu': (LgbmGpuFraudDetection, lgbm_gpu_params),
} 

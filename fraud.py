# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection_GPU.ipynb
# source: https://www.kaggle.com/dalpozz/creditcardfraud

import json
import os
import subprocess
import sys
import time

import pkg_resources
import utils
import lightgbm as lgb
from lightgbm import LGBMClassifier
from metrics import *
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
import warnings


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

    return X_train, X_test, y_train, y_test


def timeit(model, X_train, X_test, y_train, y_test):
    results = {}
    start = time.time()
    model.fit(X_train, y_train)
    results['train_time'] = time.time() - start
    start = time.time()
    y_pred = model.predict(X_test)
    results['test_time'] = time.time() - start
    results['accuracy'] = classification_metrics(y_test, y_pred)
    return results


def featurisers():
    pipeline_steps = [('scale', StandardScaler())]
    continuous_pipeline = Pipeline(steps=pipeline_steps)
    featurisers = [('continuous', continuous_pipeline)]
    return featurisers


def runXgb(X_train, X_test, y_train, y_test):
    model = Pipeline(
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
                    nthread=utils.get_number_processors()))])
    return timeit(model, X_train, X_test, y_train, y_test)


def runXgbGpu(X_train, X_test, y_train, y_test):
    params = {
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
        'updater':'grow_gpu'
    }
    return timeitXgbGpu(params, X_train, X_test, y_train, y_test)


def runXgbGpuHist(X_train, X_test, y_train, y_test):
    params = {
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
    return timeitXgbGpu(params, X_train, X_test, y_train, y_test)

def runLgbGpu(X_train, X_test, y_train, y_test):
    # data 
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

    # params
    params = {
        'num_leaves': 2**3,
        'learning_rate': 0.1,
        'scale_pos_weight': 2,
        'min_split_gain': 0.1,
        'min_child_weight': 1,
        'reg_lambda': 1,
        'subsample': 1,
        'objective':'binary',
        'task': 'train'
    }
    num_rounds = 100
    
    # train
    results = {}
    start = time.time()
    lgbm_clf_pipeline = lgb.train(params, lgb_train, num_boost_round=num_rounds)
    results['train_time'] = time.time() - start

    # predict
    start = time.time()
    y_prob_lgbm = lgbm_clf_pipeline.predict(X_test)
    results['test_time'] = time.time() - start

    y_pred_lgbm = binarize_prediction(y_prob_lgbm)

    report_lgbm = classification_metrics_binary(y_test, y_pred_lgbm)
    report2_lgbm = classification_metrics_binary_prob(y_test, y_prob_lgbm)
    report_lgbm.update(report2_lgbm)

    results['accuracy'] = report_lgbm
    return results


def timeitXgbGpu(params, X_train, X_test, y_train, y_test):
    # datasets
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # params
    num_rounds = 100

    # train
    results = dict()
    start = time.time()
    xgb_clf_pipeline = xgb.train(params, dtrain, num_boost_round=num_rounds)
    results['train_time'] = time.time() - start

    # predict
    start = time.time()
    y_prob_xgb = xgb_clf_pipeline.predict(dtest)
    results['test_time'] = time.time() - start
    y_pred_xgb = binarize_prediction(y_prob_xgb)
    results['accuracy'] = classification_metrics_binary(y_test, y_pred_xgb)
    results['accuracy'].update(classification_metrics_binary_prob(y_test, y_prob_xgb))
    return results


def runXgbHist(X_train, X_test, y_train, y_test):
    model = Pipeline(
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
                    nthread=utils.get_number_processors()))])
    return timeit(model, X_train, X_test, y_train, y_test)


def runLgb(X_train, X_test, y_train, y_test):
    model = Pipeline(
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
                    nthread=utils.get_number_processors()))])
    return timeit(model, X_train, X_test, y_train, y_test)


def benchmark(dbFolder):
    warnings.filterwarnings('ignore')
    # CPU test
    X_train, X_test, y_train, y_test = prepare(dbFolder)
    funcs = {
        'xgb':      runXgb,
        'xgb-hist': runXgbHist,
        'lgbm':     runLgb,
        'xgb-gpu': runXgbGpu,
        'xgb-gpu-hist': runXgbGpuHist,
        'lgbm-gpu': runLgbGpu,
    }
    results = {}
    for (name, func) in funcs.items():
        print("Running '%s' ..." % name)
        results[name] = func(X_train, X_test, y_train, y_test)
    
    return results

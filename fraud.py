# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/05_FraudDetection.ipynb
# source: https://www.kaggle.com/dalpozz/creditcardfraud

import json
import os
import subprocess
import sys
import time

import pkg_resources
import utils
from lightgbm import LGBMClassifier
from metrics import classification_metrics
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
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
    X_train, X_test, y_train, y_test = prepare(dbFolder)
    funcs = {
        'xgb':      runXgb,
        'xgb-hist': runXgbHist,
        'lgbm':     runLgb
    }
    results = {}
    for (name, func) in funcs.items():
        print("Running '%s' ..." % name)
        results[name] = func(X_train, X_test, y_train, y_test)
    return results

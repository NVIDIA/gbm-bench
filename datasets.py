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

import os
import sys
from enum import Enum
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module


class LearningTask(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


class Data:  # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, X_train, X_test, y_train, y_test, learning_task, qid_train=None,
                 qid_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_task = learning_task
        # For ranking task
        self.qid_train = qid_train
        self.qid_test = qid_test


def prepare_dataset(dataset_folder, dataset, nrows):
    prepare_function = globals()["prepare_" + dataset]
    return prepare_function(dataset_folder, nrows)


def prepare_airline(dataset_folder, nrows):  # pylint: disable=too-many-locals
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder,
                              "airline"
                              + ("" if nrows is None else "-" + str(nrows)) + ".pkl")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))
    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance":
            dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df = pd.read_csv(local_url,
                     names=cols, dtype=dtype_columns, nrows=nrows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
    y = df["ArrDelayBinary"]
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_bosch(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = "train_numeric.csv.zip"
    local_url = os.path.join(dataset_folder, filename)
    pickle_url = os.path.join(dataset_folder,
                              "bosch" + ("" if nrows is None else "-" + str(nrows)) + ".pkl")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    os.system("kaggle competitions download -c bosch-production-line-performance -f " +
              filename + " -p " + dataset_folder)
    X = pd.read_csv(local_url, index_col=0, compression='zip', dtype=np.float32,
                    nrows=nrows)
    y = X.iloc[:, -1]
    X.drop(X.columns[-1], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_fraud(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = "creditcard.csv"
    local_url = os.path.join(dataset_folder, filename)
    pickle_url = os.path.join(dataset_folder,
                              "creditcard" + ("" if nrows is None else "-" + str(nrows)) + ".pkl")
    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    os.system("kaggle datasets download mlg-ulb/creditcardfraud -f" +
              filename + " -p " + dataset_folder)
    df = pd.read_csv(local_url + ".zip", dtype=np.float32, nrows=nrows)
    X = df[[col for col in df.columns if col.startswith('V')]]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_higgs(dataset_folder, nrows):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder,
                              "higgs" + ("" if nrows is None else "-" + str(nrows)) + ".pkl")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    higgs = pd.read_csv(local_url, nrows=nrows)
    X = higgs.iloc[:, 1:]
    y = higgs.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data


def prepare_year(dataset_folder, nrows):
    if nrows is not None:
        print("Warning: nrows ignored for YearPredictionMSD dataset.")
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt' \
          '.zip'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    pickle_url = os.path.join(dataset_folder,
                              "year.pkl")

    if os.path.exists(pickle_url):
        return pickle.load(open(pickle_url, "rb"))

    if not os.path.isfile(local_url):
        urlretrieve(url, local_url)
    year = pd.read_csv(local_url, header=None)
    X = year.iloc[:, 1:]
    y = year.iloc[:, 0]
    # this dataset requires a specific train/test split,
    # with the specified number of rows at the start belonging to the train set,
    # and the rest being the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
                                                        train_size=463715,
                                                        test_size=51630)
    data = Data(X_train, X_test, y_train, y_test, LearningTask.REGRESSION)
    pickle.dump(data, open(pickle_url, "wb"), protocol=4)
    return data

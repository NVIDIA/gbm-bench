# Introduction
This repo tries to benchmark boosting frameworks against some of the popular
ML datasets. This is a more scriptable version of Microsoft's work on comparing
LightGBM and XGBoost: https://github.com/Azure/fast_retraining/. Most of the
datasets used here are the same as in the above repo.

# Setting up this repo
```bash
  # make sure that you have CUDA SDK (or atleast nvidia driver) installed
  # install docker and nvidia-docker, first

  # dockerfiles github project is needed to build the docker image
  $ git clone https://github.com/teju85/dockerfiles
  $ cd dockerfiles/ubuntu1604
  $ make gbm

  # this project
  $ cd ../..
  $ git clone https://github.com/teju85/gbm-perf   ## TODO!

  $ cd ..
  # Download the datasets as described in the next section.
```

# Datasets
This section contains info on how to download the datasets for this exercise.
Note that to download some datasets, one may have to sign-up in the respective
websites and probably also have to agree to their T&C's! Also note that to
maintain brevity, the dataset description is omitted out of this document.
Interested readers would want to visit respective links to know more about it.

At first, create a root folder which contains all the datasets that will be
downloaded here. Let's calls it 'gbm-datasets'. From here onwards, unless
otherwise mentioned, all folders and paths will be wrt this root folder only.
```bash
$ mkdir boosting-datasets
```

## Airline
Airline dataset and more info on it can be found here:
http://kt.ijs.si/elena_ikonomovska/data.html
```bash
$ mkdir airline
$ cd airline
$ wget http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2
```

## Allstate
This is the dataset in the Kaggle-AllState claim prediction challenge. Link for
the dataset is here: https://www.kaggle.com/c/ClaimPredictionChallenge/data
Download the 'dictionary.html' and 'train_set.zip' files from this page inside a
directory named 'allstate'.

## BCI
TODO!
Brain Computer Interaction dataset. This is something I'm not sure how the MS
researchers got from. A few trivial google-searches were unsuccessful.

## Bosch
Bosch dataset as given out on Kaggle competition here:
https://www.kaggle.com/c/bosch-production-line-performance/data. Download the
files 'train_numeric.csv.zip', 'train_date.csv.zip' and
'train_categorial.csv.zip' into a folder named 'bosch'.

## Criteo
Criteo display ad challenge as found here:
https://www.kaggle.com/c/criteo-display-ad-challenge
```bash
$ mkdir criteo
$ cd criteo
$ wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
```

## Football
Football dataset and its info can be found on Kaggle here:
https://www.kaggle.com/hugomathien/soccer. Download the 'soccer.zip' from this
page into a directory named 'football'. Code used to process this dataset is
borrowed from here:
https://www.kaggle.com/airback/match-outcome-prediction-in-football. Note that
it could take quite sometime to 'prepare' the dataset to be passed to GBM algos!

## Fraud Detection
Credit Card Fraud Detection Kaggle competition as found here:
https://www.kaggle.com/dalpozz/creditcardfraud. Download the creditcardfraud.zip
from this page into a folder named 'fraud'.

## Higgs dataset
This is the dataset as found in the UCI repo here:
https://archive.ics.uci.edu/ml/datasets/HIGGS
```bash
$ mkdir higgs
$ cd higgs
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

## IOT
Sensor stream data from Berkeley as found here:
http://www.cse.fau.edu/~xqzhu/stream.html
```bash
$ mkdir iot
$ cd iot
$ wget http://www.cse.fau.edu/~xqzhu/Stream/sensor.arff
```

## Planet: Understanding the Amazon from Space
Planet-Kaggle competition, as found here:
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data.
Download the 'train_v2.csv.zip' from this page into a directory named 'planet'.

# Running benchmarks
This assumes that one has elevated permissions on the system where this docker
image will be run for benchmarking!
```bash
  $ ./dockerfiles/scripts/launch -user gbm:latest /bin/bash
  user@container$ cd /work/gbm-perf
  user@container$ ./runme.py -root ../gbm-datasets -dataset football
  user@container$ exit
  $ cat ./gbm-perf/football.json
```

## New vs Old?
Currently, we are refactoring the benchmark logic to be easier to represent that
datasets. (there were too many classes already in the benchmarking suite!)
That's the reason one will find 2 sets of files. For eg: utils.py and
new_utils.py. Files prefixed with 'new_' are the ones that are the refactored
version of the existing files. The datasets that are currently refactored are:
* football
For these, one needs to use new_runme.py to launch benchmarks. And for the rest
use the runme.py. Soon we'll move all datasets into the new approach and will
remove this complication.

# Adding a new dataset?
Here are the steps involved in doing so:
* Assume that your dataset name is "mydataset"
* Create a folder named mydataset inside the root folder for datasets
* Copy the dataset file(s) in this folder
* Now, create a file named mydataset.py inside this repo
* Note that the name of this file is exactly the same as of the dataset!
* Create a function whose signature is 'prepareImpl(dbFolder, testSize, shuffle)'
  inside this file. dbFolder=the path to the above dataset folder; testSize=the
  size of the test-set to be created during train_test_split; shuffle=whether to
  shuffle the datapoints or not. This function should read/preprocess your
  dataset and return the 4 arrays: X_train, X_test, y_train, y_test.
* Create a another wrapper function with signature 'prepare(dbFolder)', this
  function just calls prepareImpl with hard-coded values for testSize and
  shuffle.
* Then create a map named benchmarks in the following format:
```python
benchmarks = {
    "xgb-cpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-cpu-hist": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-gpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-gpu-hist": (Enabled, BenchmarkClass, metricsFunc, params),
    "lgbm-cpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "lgbm-gpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "cat-cpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "cat-gpu": (Enabled, BenchmarkClass, metricsFunc, params),
}
# Enabled: Whether this benchmark is enabled to be run or not
# BenchmarkClass: Class to be used to instantiate and run the benchmark. For the
#                 list of classes, refer to new_utils.py
# metricsFunc: function which evaluates accuracy metrics. For the list of such
#              functions, refer to new_metrics.py
# params: map of params to be passed to the final library to customize the
#         process of training
```

# TODOs
https://yagr.nvidia.com/gradient-boosting/gbm-perf/issues

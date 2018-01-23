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
  $ make gbm80

  # this project
  $ cd ../..
  $ git clone https://github.com/teju85/gbm-bench   ## TODO!

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

## MS LTR
Microsoft's Learning to Rate query-url dataset. Download the 'MSLR-WEB10K.zip'
from the following link: https://www.microsoft.com/en-us/research/project/mslr.
Save this into a directory named 'msltr'.

## MS LTR Full
Full version of the Microsoft's Learning to Rate query-url dataset.
Download the 'MSLR-WEB30K.zip' from the following link:
https://www.microsoft.com/en-us/research/project/mslr. Save this into a
directory named 'msltr_full'.

## Planet: Understanding the Amazon from Space
Planet-Kaggle competition, as found here:
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data.
Download the 'train_v2.csv.zip' from this page into a directory named 'planet'.
Also download the 'train-jpg.tar.7z' from this page into the same directory.
Then extract the 7z file as follows:
```bash
$ 7z e train-jpg.tar.7z
```

# Benchmarking
This section assumes that one has elevated permissions on the system where this
docker image will be run for benchmarking! In case this is not true, update
your flow accordingly.

## Running a benchmark
```bash
  $ ./dockerfiles/scripts/launch -user gbm:latest /bin/bash
  user@container$ cd /work/gbm-bench
  user@container$ ./runme.py -root ../gbm-datasets -dataset football
  user@container$ exit
  $ cat ./gbm-bench/football.json
```

## Running all benchmarks and comparing results
```bash
  $ ./dockerfiles/scripts/launch -user gbm:latest /bin/bash
  user@container$ cd /work/gbm-bench
  # This generates a benchmark_5_100.csv containing runtime/perf numbers
  # This also logs all the output inside output_5_100.log
  user@container$ make MAXDEPTH=5 NTREES=100 runAll
  user@container$ ./gbm-bench/info.sh   # to get machine-info
```

# Adding a new dataset?
Here are the steps involved in doing so:
* Assume that your dataset name is "mydataset"
* Create a folder named mydataset inside the root folder for datasets
* Copy the dataset file(s) in this folder
* Now, create a file named mydataset.py inside this repo
* Note that the name of this file is exactly the same as of the dataset!
* Create a function with signature 'prepareImpl(dbFolder, testSize, shuffle)'
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
#                 list of classes, refer to utils.py
# metricsFunc: function which evaluates accuracy metrics. For the list of such
#              functions, refer to metrics.py
# params: map of params to be passed to the final library to customize the
#         process of training
```

# Adding a new library to benchmark?
Here are the steps involved in doing so:
* Open utils.py
* Create a new class there which inherits from the base class Benchmark
* Customize/Overwrite the methods as per this library's needs

# TODOs
TBD!

# Yet another boosting tree benchmark?
* This is more scriptable version (eg: for automated benchmarking)
* Also adds CatBoost to the comparison list
* Tries to keep the boosting hyper-params the same across frameworks for a fair
  comparison
* Supports multi-GPU benchmarking (assuming underlying framework allows)

# Third party codes and licensing
The third party codes which we borrowed from, and their license texts, are released
"as-received" under the folder named "3rdparty". Refer to 3rdparty/README.md as to
when they are borrowed and their respective licenses.

# License for this project
This project is released under BSD License. Refer to LICENSE for more details.

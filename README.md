# Introduction
This repo tries to benchmark boosting frameworks against some of the popular
ML datasets. This is a more scriptable version of Microsoft's work on comparing
LightGBM and XGBoost: https://github.com/Azure/fast_retraining/. Most of the
datasets used here are the same as in the above repo.

# Dependencies
- Cuda 9.2 or greater
- Nvidia docker 2.0

# Setting up this repo
```bash

  $ git clone git@gitlab.com:nvdevtech/gbm-bench.git
  # The below link will be made active once we open-source this work
  # $ git clone https://github.com/NVIDIA/gradient-boosted-benchmarks
  $ cd gbm-bench
  $ docker build -t gbm-bench:9.2 .

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
$ mkdir gbm-datasets
```

## Airline
Airline dataset and more info on it can be found here:
http://kt.ijs.si/elena_ikonomovska/data.html
```bash
$ mkdir airline
$ cd airline
$ wget http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2
```

## Bosch
Bosch dataset as given out on Kaggle competition here:
https://www.kaggle.com/c/bosch-production-line-performance/data. Download the
files 'train_numeric.csv.zip', 'train_date.csv.zip' and
'train_categorial.csv.zip' into a folder named 'bosch'.

## Criteo CTR (> 1 TB)
Criteo CTR (click-through rate) data as found here:
http://labs.criteo.com/2013/12/download-terabyte-click-logs-2/
```bash
$ mkdir criteo
$ cd criteo
$ for day in `seq 0 23`; do curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_$day.gz; done
```
(ETL instructions to come)

## Fraud Detection
Credit Card Fraud Detection Kaggle competition as found here:
https://www.kaggle.com/mlg-ulb/creditcardfraud. Download the creditcardfraud.zip
from this page into a folder named 'fraud'.

## Higgs dataset
This is the dataset as found in the UCI repo here:
https://archive.ics.uci.edu/ml/datasets/HIGGS
```bash
$ mkdir higgs
$ cd higgs
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
```

## Mortgage
TBD: add information

## Mortgage Data from Fannie Mae
Download the raw acquisition and performance data from here:
http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
(ETL instructions to come)

## MS LTR
Microsoft's Learning to Rate query-url dataset. Download the 'MSLR-WEB10K.zip'
from the following link: https://www.microsoft.com/en-us/research/project/mslr.
Save this into a directory named 'msltr'.

# Benchmarking
This section assumes that one has elevated permissions on the system where this
docker image will be run for benchmarking! In case this is not true, update
your flow accordingly.

## Launching container
```bash
docker run --runtime=nvidia -it --rm -v {YOUR-LOCATION/gbm-datasets}:/opt/gbm-datasets -v gbm-bench:/opt/gbm-bench gbm-bench:9.2 /bin/bash
```
Basically, make sure that you have mounted the datasets directory inside the container.

## Running a dataset
```bash
  user@container$ cd /opt/gbm-bench
  user@container$ ./runme.py -root ../gbm-datasets -dataset football
  user@container$ cat ./gbm-bench/football.json
```

## Running all datasets and benchmarks to compare results
```bash
  user@container$ cd /opt/gbm-bench
  # This generates a benchmark_5_100.csv containing runtime/perf numbers
  # This also logs all the output inside output_5_100.log
  user@container$ make MAXDEPTH=5 NTREES=100 runAll
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
    "xgb-cpu-exact": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-cpu": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-gpu-exact": (Enabled, BenchmarkClass, metricsFunc, params),
    "xgb-gpu": (Enabled, BenchmarkClass, metricsFunc, params),
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
* Add a line inside Makefile under '_runAll' target to benchmark this particular
  dataset as well.

# Trouble shooting
## [LightGBM] [Warning] boost::filesystem::create_directories: Permission denied: ...
```bash
export BOOST_COMPUTE_USE_OFFLINE_CACHE=0
```
Reference: Issue [here](https://github.com/Microsoft/LightGBM/issues/1531)

## Warning! catboost sets max-thread-count to 56!
Pass a `-cpus 56` option to the 'runme.py'

# Adding a new library to benchmark?
Here are the steps involved in doing so:
* Open utils.py
* Create a new class there which inherits from the base class Benchmark
* Customize/Overwrite the methods as per this library's needs

# Yet another boosting tree benchmark?
* This is more scriptable (and configurable) version (eg: for automated benchmarking)
* Also adds CatBoost to the comparison list
* Tries to keep the boosting hyper-params the same across frameworks for a fair
  comparison. Reference: [this paper](https://openreview.net/pdf?id=ryexWdLRtm)
* Supports multi-GPU as well as multi-node benchmarking (assuming underlying framework allows)

# Third party codes and licensing
The third party codes which we borrowed from, and their license texts, are released
"as-received" under the folder named "3rdparty". Refer to 3rdparty/README.md as to
when they are borrowed and their respective licenses.

# License for this project
This project is released under BSD License. Refer to LICENSE for more details.

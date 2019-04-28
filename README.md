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
```

# Datasets
gbm-bench will automatically download datasets as needed using wget or the [Kaggle API](https://github.com/Kaggle/kaggle-api). To use the kaggle datasets you will need a valid kaggle account and API token. Create a folder 'gbm-datasets' in some location with sufficient space for large datasets. Mounting this folder on fast local storage as opposed to network storage is recommended.

```bash
$ mkdir gbm-datasets
```
Upon launching docker you will pass this folder as well as the location of the kaggle API key as volumes to the container.

| Name                                                                           | Rows   | Columns | Task           |
|--------------------------------------------------------------------------------|--------|---------|----------------|
| [airline](http://kt.ijs.si/elena_ikonomovska/data.html)                        | 115M   | 13      | Classification |
| [bosch](https://www.kaggle.com/c/bosch-production-line-performance)            | 1.184M | 968     | Classification |
| [fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)                        | 285K   | 28      | Classification |
| [higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS)                         | 11M    | 28      | Classification |
| [year](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd)              | 515K   | 90      | Regression     |
| [covtype](https://archive.ics.uci.edu/ml/datasets/covertype)                   | 581K   | 54      | Multiclass     |
| [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) | 500K   | 2000    | Classification |

# Benchmarking
This section assumes that one has elevated permissions on the system where this
docker image will be run for benchmarking! In case this is not true, update
your flow accordingly.

## Launching container
```bash
docker run --runtime=nvidia -it --rm \
    -w /opt/gbm-bench \
    -v {YOUR-LOCATION/gbm-datasets}:/opt/gbm-datasets \
    -v {YOUR-LOCATION/gbm-bench}:/opt/gbm-bench \
    -v {KAGGLE-API-LOCATION/.kaggle}:/root/.kaggle \
     gbm-bench:9.2 /bin/bash
```
The above command launches an interactive session and mounts the dataset folder, the gbm-bench repo and your kaggle API key inside the container.

## Running benchmarks
Benchmarks are launched from the python runme.py script
```bash
python runme.py --help
usage: runme.py [-h] [-dataset DATASET] [-root ROOT] [-algorithm ALGORITHM]
                [-gpus GPUS] [-cpus CPUS] [-output OUTPUT] [-ntrees NTREES]
                [-nrows NROWS] [-warmup] [-verbose] [-extra EXTRA]

Benchmark xgboost/lightgbm/catboost on real datasets

optional arguments:
  -h, --help            show this help message and exit
  -dataset DATASET      The dataset to be used for benchmarking. 'all' for all
                        datasets.
  -root ROOT            The root datasets folder
  -algorithm ALGORITHM  Comma-separated list of algorithms to run; 'all' run
                        all
  -gpus GPUS            #GPUs to use for the benchmarks; ignored when not
                        supported. Default is to use all.
  -cpus CPUS            #CPUs to use for the benchmarks; 0 means
                        psutil.cpu_count(logical=False)
  -output OUTPUT        Output json file with runtime/accuracy stats
  -ntrees NTREES        Number of trees. Default is as specified in the
                        respective dataset configuration
  -nrows NROWS          Subset of rows in the datasets to use. Useful for test
                        running benchmarks on small amounts of data. WARNING:
                        Some datasets will give incorrect accuracy results if
                        nrows is specified as they have predefined train/test
                        splits.
  -warmup               Whether to run a small benchmark (fraud) as a warmup
  -verbose              Produce verbose output
  -extra EXTRA          Extra arguments as a python dictionary
```

As an example, launch the xgb-gpu algorithm on the year dataset.
```bash
python runme.py -dataset year -algorithm xgb-gpu
```
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

# Introduction
This page tries to document the steps to be followed while trying to benchmark
a "bleeding-edge" xgboost repository. It assumes that the user is interested in
benchmarking *only* the xgboost framework and not others.

# Initial setup (one-time)
In the below set of commands, replace the "copy" command to setup your dataset
directory with the right dataset of your interest, in case it is not mortgage.
```bash
$ mkdir /raid/gtc-18
$ cd /raid/gtc-18
$ git clone https://github.com/teju85/dockerfiles
$ git clone https://gitlab.com/nvcollab/gbm-bench
$ git clone --recursive https://gitlab.com/nvcollab/nv-xgboost
$ cd nv-xgboost
$ git checkout --track -b gtc18 origin/gtc18  # this is needed only until GTC!
$ mkdir gbm-datasets
$ cp -r /raid/mortgage/output_np_binary/ gbm-datasets/mortgage
$ cd dockerfiles/ubuntu1604
$ make xgb-dev90
$ cd ../..
```

# Launching the container (and benchmarking)
## Launching the container
```bash
$ ./dockerfiles/scripts/launch -runas user xgb:dev-9.0 /bin/bash
```
Everything that follows in this section assume that you are running out of the
docker container started above.

## Building xgboost
It is recommended to run this command, every time you start a new container!
```bash
$ cd /work/nv-xgboost
$ mkdir build-9.0
$ cd build-9.0
$ cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON
$ make -j8
$ export PYTHONPATH=`pwd`:`pwd`/../python-package
$ touch catboost.py lightgbm.py
```
Unless and otherwise mentioned, from here onwards, in this section, it is assumed
that your current directory is always "/work/nv-xgboost/build-9.0".

## Running benchmarks
We will assume that you are benchmarking the mortgage dataset here. If the dataset
is different than this, replace all its occurences with yours.
```bash
$ ../../gtc-benchmark.sh
```
Currently, this script can be located inside "/raid/gtc-18" folder, on dgx21 node.
Plan is to put it inside gbm-bench for others to view as well.


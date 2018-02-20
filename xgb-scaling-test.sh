#!/bin/bash
#
# Description:
# Performs xgboost scalability study on multi-gpu's.
#
# Pre-reqs:
#  . DGX box
#  . pwd = <xgboostRootDir>/build
#  . Xgboost has been built with "cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON"
#  . <xgboostRootDir> and gbm-bench are in the same directory
#
# Usage:
#  ../../gbm-bench/xgb-scaling-test.sh
#
# Output:
# Prints a csv formatted table of training/testing times are accuracy metrics
# to the stdout, as a function of number of GPUs and also a function of whether
# cpu or gpu based prediction was used
#
dataset=${1:-airline_ext}
export PYTHONPATH=`pwd`:`pwd`/../python-package
touch catboost.py lightgbm.py
for ngpus in 1 2 4 8; do
    for pred in cpu gpu; do
        ../../gbm-bench/runme.py -root ../../gbm-datasets/ \
                                 -dataset $dataset \
                                 -benchmark xgb-gpu-hist \
                                 -ngpus $ngpus \
                                 -extra "{'predictor':'${pred}_predictor'}" \
                                 -output airline_ext_${pred}_${ngpus}.json
    done
done
../../gbm-bench/json2csv.py *.json

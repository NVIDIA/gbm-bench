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
# to the stdout, as a function of number of GPUs
#

function cudaVisibleDevices() {
    local ngpus=$1
    local isvolta=$2
    # in case of DGX1v, there are gpu pairs with double connections between them
    # and that's because every GPU has 6 nvlink connections as opposed to 4 in
    # DGX1p
    if [ "$dgx1v" = "1" ]; then
        case $ngpus in
            1)
                echo "0";;
            2)
                echo "0,3";;
            4)
                echo "0,1,2,3";;
            8)
                echo "0,3,2,1,5,6,7,4";;
        esac
    else
        case $ngpus in
            1)
                echo "0";;
            2)
                echo "0,2";;
            4)
                echo "0,1,2,3";;
            8)
                echo "0,1,2,3,4,5,6,7";;
        esac
    fi
}

dgx1v=${1:-0}
ncpus=40
dataset=airline_ext
export PYTHONPATH=`pwd`:`pwd`/../python-package
touch catboost.py lightgbm.py
for ngpus in 1 2 4 8; do
    devs=`cudaVisibleDevices $ngpus $dgx1v`
    env CUDA_VISIBLE_DEVICES=$devs \
        OMP_PROC_BIND=TRUE \
        OMP_NUM_THREADS=$ncpus \
        OMP_PLACES="{0}:$ncpus" \
        OMP_DISPLAY_ENV=TRUE \
        ../../gbm-bench/runme.py -root ../../gbm-datasets/ \
            -dataset $dataset \
            -benchmark xgb-gpu-hist \
            -ngpus $ngpus \
            -ncpus $ncpus \
            -extra "{'predictor':'gpu_predictor', 'debug_verbose':1}" \
            -output ${dataset}_${ngpus}.json
done
../../gbm-bench/json2csv.py *.json

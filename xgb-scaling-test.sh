#!/bin/bash
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

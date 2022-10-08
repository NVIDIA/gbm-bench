#!/usr/bin/env python
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import argparse
import json
import ast
import psutil
import datetime
import algorithms
from metrics import get_metrics
from datasets import prepare_dataset


def get_number_processors(args):
    if args.cpus == 0:
        return psutil.cpu_count(logical=False)
    return args.cpus


def print_sys_info(args):
    try:
        import xgboost  # pylint: disable=import-outside-toplevel
        print("Xgboost : %s" % xgboost.__version__)
    except ImportError:
        pass
    try:
        import lightgbm  # pylint: disable=import-outside-toplevel
        print("LightGBM: %s" % lightgbm.__version__)
    except (ImportError, OSError):
        pass
    try:
        import catboost  # pylint: disable=import-outside-toplevel
        print("Catboost: %s" % catboost.__version__)
    except ImportError:
        pass
    print("System  : %s" % sys.version)
    print("#jobs   : %d" % args.cpus)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark xgboost/lightgbm/catboost on real datasets")
    parser.add_argument("-dataset", default="all", type=str,
                        help="The dataset to be used for benchmarking. 'all' for all datasets.")
    parser.add_argument("-root", default="/opt/gbm-datasets",
                        type=str, help="The root datasets folder")
    parser.add_argument("-algorithm", default="all", type=str,
                        help=("Comma-separated list of algorithms to run; "
                              "'all' run all"))
    parser.add_argument("-gpus", default=-1, type=int,
                        help=("#GPUs to use for the benchmarks; "
                              "ignored when not supported. Default is to use all."))
    parser.add_argument("-cpus", default=0, type=int,
                        help=("#CPUs to use for the benchmarks; "
                              "0 means psutil.cpu_count(logical=False)"))
    parser.add_argument("-output", default=sys.path[0] + "/results.json", type=str,
                        help="Output json file with runtime/accuracy stats")
    parser.add_argument("-ntrees", default=500, type=int,
                        help=("Number of trees. Default is as specified in "
                              "the respective dataset configuration"))
    parser.add_argument("-nrows", default=None, type=int,
                        help=(
                            "Subset of rows in the datasets to use. Useful for test running "
                            "benchmarks on small amounts of data. WARNING: Some datasets will "
                            "give incorrect accuracy results if nrows is specified as they have "
                            "predefined train/test splits."))
    parser.add_argument("-warmup", action="store_true",
                        help=("Whether to run a small benchmark (fraud) as a warmup"))
    parser.add_argument("-verbose", action="store_true", help="Produce verbose output")
    parser.add_argument("-extra", default='{}', help="Extra arguments as a python dictionary")
    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "%s.json" % args.dataset
    return args


# benchmarks a single dataset
def benchmark(args, dataset_folder, dataset):
    data = prepare_dataset(dataset_folder, dataset, args.nrows)
    results = {}
    # "all" runs all algorithms
    if args.algorithm == "all":
        args.algorithm = "xgb-gpu,xgb-cpu,xgb-gpu-dask,lgbm-cpu,lgbm-gpu,cat-cpu,cat-gpu"
    for alg in args.algorithm.split(","):
        print("Running '%s' ..." % alg)
        runner = algorithms.Algorithm.create(alg)
        with runner:
            train_time = runner.fit(data, args)
            pred = runner.test(data)
            results[alg] = {
                "train_time": train_time,
                "accuracy": get_metrics(data, pred),
            }

    return results


def main():
    args = parse_args()
    args.cpus = get_number_processors(args)
    args.extra = ast.literal_eval(args.extra)
    print_sys_info(args)
    ts=datetime.datetime.utcnow().strftime('%Y%m%d.%H%M%S%f')
    if args.warmup:
        benchmark(args, os.path.join(args.root, "fraud"), "fraud")
    if args.dataset == 'all':
        args.dataset = 'airline,bosch,fraud,higgs,year,epsilon,covtype,newsgroups'
    results = {}
    for dataset in args.dataset.split(","):
        folder = os.path.join(args.root, dataset)
        results.update({ 'timestamp_utc': ts,
                         dataset: benchmark(args, folder, dataset)})
        print(json.dumps({dataset: results[dataset]}, indent=2, sort_keys=True))
    output = json.dumps(results, indent=2, sort_keys=True)
    output_file = open(args.output, "w")
    output_file.write(output + "\n")
    output_file.close()
    print("Results written to file '%s'" % args.output)


if __name__ == "__main__":
    main()

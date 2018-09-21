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

import ast
import os
import sys
import argparse
import utils
import json
import warnings

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Benchmark xgboost/lightgbm/catboost on real datasets")
    parser.add_argument("-dataset", default="football", type=str,
                        help="The dataset to be used for benchmarking")
    parser.add_argument("-root", default="/datasets",
                        type=str, help="The root datasets folder")
    parser.add_argument("-benchmarks", default="all", type=str,
                        help=("Comma-separated list of benchmarks to run; "
                              "'all' run all"))
    parser.add_argument("-ngpus", default=1, type=int,
                        help=("#GPUs to use for the benchmarks; "
                              "ignored when not supported"))
    parser.add_argument("-ncpus", default=0, type=int,
                        help=("#CPUs to use for the benchmarks; "
                              "0 means multiprocessing.cpu_count()"))
    parser.add_argument("-output", default=None, type=str,
                        help="Output json file with runtime/accuracy stats")
    parser.add_argument("-maxdepth", default=None, type=int,
                        help=("Max-depth of trees. Default is as specified in "
                              "the respective dataset configuration"))
    parser.add_argument("-ntrees", default=None, type=int,
                        help=("Number of trees. Default is as specified in "
                              "the respective dataset configuration"))
    parser.add_argument("-nrows", default=None, type=int,
                        help=("Total rows to be used for training/test. "
                              "Default is to consume full dataset. However, "
                              "for this to work, the dataset module should "
                              "support customizing rows. Currently only "
                              "airline and airline_ext do so!"))
    parser.add_argument("-extra", default='{}',
                        help="Extra arguments as a python dictionary")
    parser.add_argument("-warmup", action="store_true",
                        help=("Whether to run a small benchmark (fraud) as a warmup"))
    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "%s.json" % args.dataset
    return args

# add extra parameters for benchmarks (if present)
def addExtraParams(params, extraParams, bName):
    # multi gpu case
    if "n_gpus" in extraParams:
        if "xgb" in bName or "rf" in bName:
            params["n_gpus"] = extraParams["n_gpus"]
        else:
            print("'n_gpus' currently only applies to 'xgboost'")
    # if need to customize tree depth
    if "maxdepth" in extraParams:
        if "xgb" in bName or "rf" in bName:
            params["max_depth"] = extraParams["maxdepth"]
            params["max_leaves"] = 2**extraParams["maxdepth"]
        elif "lgbm" in bName:
            params["num_leaves"] = 2**extraParams["maxdepth"]
        elif "cat" in bName:
            params["depth"] = extraParams["maxdepth"]
    # if need to customize number of boosters
    if "ntrees" in extraParams:
        if "xgb" in bName or "rf" in bName or "lgbm" in bName:
            params["num_round"] = extraParams["ntrees"]
        elif "cat" in bName:
            params["iterations"] = extraParams["ntrees"]
    # if need to pass other parameters directly to the benchmark
    if "extra" in extraParams:
        params.update(extraParams["extra"])
    return

# benchmarks a single dataset
def benchmark(dbFolder, module, benchmarks, extra_params, nrows):
    warnings.filterwarnings("ignore")
    data = module.prepare(dbFolder, nrows)
    funcs = module.benchmarks
    results = {}
    # "all" runs all benchmarks
    if benchmarks[0] == "all":
        benchmarks = funcs.keys()
    for name in benchmarks:
        enabled, cls, metrics, params = funcs[name]
        params = params.copy()
        if not enabled:
            print("Skipping '%s'... " % name)
            continue
        addExtraParams(params, extra_params, name)
        print("Running '%s' ..." % name)

        runner = cls(data, params)
        with runner:
            (prepare_time, train_time, test_time) = runner.run()
            y_test = runner.y_test_matrix()
            y_pred = runner.y_pred
            #print(type(y_test))
            #print(type(y_pred))
            results[name] = {
                "prepare_time": prepare_time,
                "train_time": train_time,
                "test_time":  test_time,
                "accuracy":   metrics(y_test, y_pred),
            }

    return results

def main():
    args = parseArgs()
    if args.ncpus > 0:
        utils.number_processors_override = args.ncpus
    utils.print_sys_info()
    folder = os.path.join(args.root, args.dataset)
    benchmarks = args.benchmarks.split(",")
    # TODO: this is a HACK to support dynamic loading of modules at runtime!
    module = __import__(args.dataset)
    extra_params = {"n_gpus": args.ngpus}
    if args.maxdepth is not None:
        extra_params["maxdepth"] = args.maxdepth
    if args.ntrees is not None:
        extra_params["ntrees"] = args.ntrees
    extra_params["extra"] = ast.literal_eval(args.extra)
    if args.warmup:
        warmup_extra_params = {"n_gpus": args.ngpus}
        benchmark(os.path.join(args.root, "fraud"), __import__("fraud"),
                  benchmarks, warmup_extra_params, args.nrows)
    results = benchmark(folder, module, benchmarks, extra_params, args.nrows)
    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    fp = open(args.output, "w")
    fp.write(output + "\n")
    fp.close()
    print("Results written to file '%s'" % args.output)

if __name__ == "__main__":
    main()

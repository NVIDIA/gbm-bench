#!/usr/bin/env python

import os
import sys
import argparse
import utils
import json
import warnings

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Benchmark xgboost/lightgbm on real datasets")
    parser.add_argument("-dataset", default="football", type=str,
                        help="The dataset to be used for benchmarking")
    parser.add_argument("-root", default="/datasets",
                        type=str, help="The root datasets folder")
    parser.add_argument(
        "-benchmarks", default="all", type=str,
        help="The comma-separated list of benchmarks to run; 'all' run all of them")
    parser.add_argument(
        "-ngpus", default=1, type=int,
        help=("The number of GPUs to use for the benchmarks; "
              "ignored when not supported"))
    args = parser.parse_args()
    return args

# benchmarks a single dataset
def benchmark(dbFolder, module, benchmarks, extra_params):
    warnings.filterwarnings('ignore')
    data = module.prepare(dbFolder)
    funcs = module.benchmarks
    results = {}
    # 'all' runs all benchmarks
    if benchmarks[0] == 'all':
        benchmarks = funcs.keys()
    for name in benchmarks:
        cls, params = funcs[name]
        # add extra parameters for params-based benchmarks
        if type(params) is dict:
            for (extra_key, extra_value) in extra_params.items():
                # only set n_gpus for xgboost,
                # as LightGBM does not support it
                # this currently only affects xgb-gpu-hist benchmarks
                if extra_key == 'n_gpus' and 'xgb' not in name:
                    continue
                params[extra_key] = extra_value
        print("Running '%s' ..." % name)
        results[name] = cls(data, params).run()
    
    return results

def main():
    args = parseArgs()
    utils.print_sys_info()
    folder = os.path.join(args.root, args.dataset)
    benchmarks = args.benchmarks.split(',')
    # TODO: this is a HACK to support dynamic loading of modules at runtime!
    module = __import__(args.dataset)
    extra_params = {'n_gpus': args.ngpus}
    results = benchmark(folder, module, benchmarks, extra_params)
    print(json.dumps(results, indent=2, sort_keys=True))
    return

if __name__ == '__main__':
    main()

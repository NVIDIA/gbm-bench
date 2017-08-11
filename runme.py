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
    args = parser.parse_args()
    return args

# benchmarks a single dataset
def benchmark(dbFolder, module):
    warnings.filterwarnings('ignore')
    data = module.prepare(dbFolder)
    funcs = module.benchmarks
    results = {}
    for (name, class_params) in funcs.items():
        cls, params = class_params
        print("Running '%s' ..." % name)
        results[name] = cls(data, params).run()
    
    return results

def main():
    args = parseArgs()
    utils.print_sys_info()
    folder = os.path.join(args.root, args.dataset)
    # TODO: this is a HACK to support dynamic loading of modules at runtime!
    module = __import__(args.dataset)
    results = benchmark(folder, module)
    print(json.dumps(results, indent=2, sort_keys=True))
    return

if __name__ == '__main__':
    main()

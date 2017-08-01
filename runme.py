#!/usr/bin/env python

import os
import sys
import argparse
import utils
import json

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Benchmark xgboost/lightgbm on real datasets")
    parser.add_argument("-dataset", default="football", type=str,
                        help="The dataset to be used for benchmarking")
    parser.add_argument("-root", default="/datasets",
                        type=str, help="The root datasets folder")
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    utils.print_sys_info()
    folder = os.path.join(args.root, args.dataset)
    # TODO: this is a HACK to support dynamic loading of modules at runtime!
    results = __import__(args.dataset).benchmark(folder)
    print(json.dumps(results, indent=2, sort_keys=True))
    return

if __name__ == '__main__':
    main()

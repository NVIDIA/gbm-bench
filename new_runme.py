#!/usr/bin/env python
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
    args = parser.parse_args()
    return args

# benchmarks a single dataset
def benchmark(dbFolder, module, benchmarks, extra_params):
    warnings.filterwarnings("ignore")
    data = module.prepare(dbFolder)
    funcs = module.benchmarks
    results = {}
    # "all" runs all benchmarks
    if benchmarks[0] == "all":
        benchmarks = funcs.keys()
    for name in benchmarks:
        enabled, cls, metrics, params = funcs[name]
        if not enabled:
            print("Skipping '%s'... " % name)
            continue
        # add extra parameters for benchmarks (if present)
        for (extra_key, extra_value) in extra_params.items():
            # only set n_gpus for xgboost,
            # as LightGBM/CatBoost do not support it
            # this currently only affects xgb-gpu-hist benchmarks
            if extra_key == "n_gpus" and "xgb" not in name:
                continue
            params[extra_key] = extra_value
        print("Running '%s' ..." % name)
        runner = cls(data, params)
        (train_time, test_time) = runner.run()
        results[name] = {
            "train_time": train_time,
            "test_time":  test_time,
            "accuracy":   metrics(runner.data.y_test, runner.y_pred),
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
    results = benchmark(folder, module, benchmarks, extra_params)
    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    fp = open("%s.json" % args.dataset, "w")
    fp.write(output + "\n")
    fp.close()

if __name__ == "__main__":
    main()

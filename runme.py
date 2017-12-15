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
    parser.add_argument("-output", default=None, type=str,
                        help="Output json file with runtime/accuracy stats")
    parser.add_argument("-maxdepth", default=None, type=int,
                        help=("Max-depth of trees. Default is as specified in "
                              "the respective dataset configuration"))
    parser.add_argument("-ntrees", default=None, type=int,
                        help=("Number of trees. Default is as specified in "
                              "the respective dataset configuration"))
    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "%s.json" % args.dataset
    return args

# add extra parameters for benchmarks (if present)
def addExtraParams(params, extraParams, bName):
    # multi gpu case
    if "n_gpus" in extraParams:
        if "xgb" in bName:
            params[extra_key] = extra_value
        else:
            print("'n_gpus' currently only applies to 'xgboost'")
    # if need to customize tree depth
    if "maxdepth" in extraParams:
        if "xgb" in bName:
            params["max_depth"] = extraParams["maxdepth"]
            params["max_leaves"] = 2**extraParams["maxdepth"]
        elif "lgbm" in bName:
            params["num_leaves"] = 2**extraParams["maxdepth"]
        elif "cat" in bName:
            params["depth"] = extraParams["maxdepth"]
    # if need to customize number of boosters
    if "ntrees" in extraParams:
        if "xgb" in bName or "lgbm" in bName:
            params["num_round"] = extraParams["ntrees"]
        elif "cat" in bName:
            params["iterations"] = extraParams["ntrees"]
    return

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
        addExtraParams(params, extra_params, name)
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
    if args.maxdepth is not None:
        extra_params["maxdepth"] = args.maxdepth
    if args.ntrees is not None:
        extra_params["ntrees"] = args.ntrees
    results = benchmark(folder, module, benchmarks, extra_params)
    output = json.dumps(results, indent=2, sort_keys=True)
    print(output)
    fp = open(args.output, "w")
    fp.write(output + "\n")
    fp.close()
    print("Results written to file '%s'" % args.output)

if __name__ == "__main__":
    main()

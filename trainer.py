import os
import sys
import argparse
import json
import ast
import psutil
import algorithms
from metrics import get_metrics
from runme import benchmark
from runme import get_number_processors
from runme import print_sys_info
from datasets import prepare_dataset

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
    parser.add_argument("-ngpus", default='1', type=str,
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
    parser.add_argument("-cycles", default=1, type=int,
                        help=("#training of training cycles for each iteration"))
    parser.add_argument("-train_cycles", default=1, type=int,
                        help=("#training of training cycles"))
    parser.add_argument("-warmup", action="store_true",
                        help=("Whether to run a small benchmark (fraud) as a warmup"))
    parser.add_argument("-verbose", action="store_true", help="Produce verbose output")
    parser.add_argument("-extra", default='{}', help="Extra arguments as a python dictionary")
    args = parser.parse_args()
    # default value for output json file
    if not args.output:
        args.output = "%s.json" % args.dataset
    return args

def main():
    args = parse_args()
    args.cpus = get_number_processors(args)
    args.extra = ast.literal_eval(args.extra)
    print_sys_info(args)
    gpu_lis = []
    if args.warmup:
        benchmark(args, os.path.join(args.root, "fraud"), "fraud")
    if args.dataset == 'all':
        args.dataset = 'airline,bosch,fraud,higgs,year,epsilon,covtype'
    gpu_lis = args.ngpus.split(",")
    if len(gpu_lis) != args.train_cycles:
        print("please match npus with train_cycles")
    else:
        for idx, ele in enumerate(range(args.train_cycles)):
            results = {}
            args.gpus = int(gpu_lis[idx])
            for dataset in args.dataset.split(","):
                folder = os.path.join(args.root, dataset)
                results.update({dataset: benchmark(args, folder, dataset)})
                print(json.dumps({dataset: results[dataset]}, indent=2, sort_keys=True))
            output = json.dumps(results, indent=2, sort_keys=True)
            output_file = open(args.output[:-5]+str(idx)+args.output[-5:], "w")
            output_file.write(output + "\n")
            output_file.close()
            print("Results written to file '%s'" % args.output)

if __name__ == '__main__':
    main()
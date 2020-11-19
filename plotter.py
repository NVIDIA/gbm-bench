import os
import sys
import csv
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import json2csv
import csv_merger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize benchmarks against another version")
    parser.add_argument("-d1", required=True, type=str,
                        help="comma sperated csv files to merge")
    parser.add_argument("-metric", default="train_time", type=str,
                        help=("The metric we want to visulaize"))
    parser.add_argument("-dataset", required=True, type=str,
                        help="dataset to plot")               
    parser.add_argument("-title", default="graph", type=str,
                        help=("The title of the graph"))
    parser.add_argument("-output", default=sys.path[0] + "/results.png", type=str,
                        help="Output json file with visualization")
    args = parser.parse_args()
    return args

def plot_error_bars(df_lis, args):
    fig, ax = plt.subplots()
    labels = ["xgb", "cat"]
    ngpu = [1, 2, 4, 6, 8]
    for idx,df in enumerate(df_lis):
        gp = df.groupby("dataset")
        means = gp.mean()
        errors = gp.std()
        means = means.T
        means = means.drop(columns=[x for x,y in means.iteritems() if x != args.dataset])
        plt.plot(ngpu, means[args.dataset].tolist(), label=labels[idx])
        ax.legend()
        plt.ylabel(args.metric)
        plt.xticks()
        plt.title(f"{args.title} plot") 
    plt.savefig(args.output)

def main():
    df_lis = []
    args = parse_args()
    groups = args.d1.split(":")
    for x in groups:
        df_lis.append(csv_merger.import_main(x))
    plot_error_bars(df_lis, args)

if __name__ == '__main__':
    main()
import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize benchmarks against another version")
    parser.add_argument("-d1", required=True, type=str,
                        help="The first csv file directory with results")
    parser.add_argument("-d2", required=True, type=str,
                        help="The second csv file directory with results")
    parser.add_argument("-metric", default="train_time", type=str,
                        help=("The metric we want to visulaize"))
    parser.add_argument("-title", default="graph", type=str,
                        help=("The title of the graph"))
    parser.add_argument("-output", default=sys.path[0] + "/results.png", type=str,
                        help="Output json file with visualization")
    args = parser.parse_args()
    return args

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def algo_trimmer(algo):
    print(f"algo: {algo}")
    s = ""
    for i in range(len(algo)):
        if(algo[i] == '_'):
            break
        s+=algo[i]
    return s

def plot_error_bars(df, args):
    gp = df.groupby("dataset")
    means = gp.mean()
    errors = gp.std()
    if(args.metric == "train-time"):
        fig, ax = plt.subplots()
        pl = means.plot.bar(yerr=errors, ax=ax, capsize=10, rot=0, logy=True)
        for idx, label in enumerate(list(means.index)): 
            for acc in means.columns:
                value = np.round(means.iloc[idx,0]/means.iloc[idx, 1],decimals=2)
                ax.annotate(value,
                            (idx, value),
                            xytext=(0, 15), 
                            textcoords='offset points')
    else:
        fig, ax = plt.subplots()
        pl = means.plot.bar(yerr=errors, ax=ax, capsize=10, rot=0, figsize=(12, 12))
        for p in ax.patches:
            h = p.get_height()
            x = p.get_x()+p.get_width()/2.
            if h != 0:
                ax.annotate("%g" % round(p.get_height(), 2), xy=(x,h), xytext=(0,4), rotation=90,
                   textcoords="offset points", ha="center", va="bottom")
        ax.legend(ncol=len(df.columns), loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), 
          borderaxespad=0, mode="expand")
    plt.ylabel(f"{args.metric}")
    plt.title(args.title) 
    plt.savefig(args.output)

def train_timer(args):
    df = pd.DataFrame()
    for idx, i in enumerate([args.d1, args.d2]):
        temp_df = pd.read_csv(i)
        temp_df = temp_df.drop(columns=["test_time", "AUC", "Accuracy", "F1","Precision",
        "Recall","MeanAbsError","MeanSquaredError","MedianAbsError", "algorithm"])
        if(idx == 0):
            temp_df = temp_df.rename(columns={"train_time": f"{args.d1}_train_time"})
            df = df.append(temp_df)
        else:
            temp_df = temp_df.rename(columns={"train_time": f"{args.d2}_train_time"})
            df = df.merge(temp_df, on="dataset")
    return df

def accuracy_timer(args):
    df = pd.DataFrame()
    for idx, i in enumerate([args.d1, args.d2]):
        temp_df = pd.read_csv(i)
        temp_df = temp_df.drop(columns=["test_time", "AUC", "train_time", "F1","Precision",
        "Recall","MeanAbsError","MeanSquaredError","MedianAbsError", "algorithm"])
        if(idx == 0):
            temp_df = temp_df[temp_df["Accuracy"] != "-na-"]
            temp_df = temp_df.rename(columns={"Accuracy": f"{args.d1}_Accuracy"})
            df = df.append(temp_df)
        else:
            temp_df = temp_df[temp_df["Accuracy"] != "-na-"]
            temp_df = temp_df.rename(columns={"Accuracy": f"{args.d2}_Accuracy"})
            df = df.merge(temp_df, on="dataset")
    df = df.astype({f"{args.d1}_Accuracy": 'float32', f"{args.d2}_Accuracy": 'float32'})
    return df

def main():
    args = parse_args()
    df = pd.DataFrame()
    if(args.metric == 'train-time'):
        df = train_timer(args)
    else:
        df = accuracy_timer(args)
    plot_error_bars(df, args)


if __name__ == '__main__':
    main()
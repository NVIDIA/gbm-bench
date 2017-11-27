#!/usr/bin/env python

import sys
import json
import string
import os
import csv

Datasets = ["airline", "bosch", "football", "fraud", "higgs", "msltr",
            "msltr_full", "planet"]
Algos = ["cat-cpu", "cat-gpu", "lgbm-cpu", "lgbm-gpu", "xgb-cpu", "xgb-gpu",
         "xgb-cpu-hist", "xgb-gpu-hist"]
Timings = ["train_time", "test_time"]
Metrics = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
AllMetrics = Timings + Metrics

def loadPerfData(jsonFile):
    fp = open(jsonFile, "r")
    data = json.load(fp)
    fp.close()
    return data

def loadAllPerfData(files):
    data = {}
    for jsonFile in sys.argv[1:]:
        dataset = os.path.basename(jsonFile)
        dataset = string.replace(dataset, ".json", "")
        data[dataset] = loadPerfData(jsonFile)
    return data

def readFromDict(hashmap, key, defVal="-na-"):
    d = hashmap[key] if key in hashmap else defVal
    return d

def combinePerfData(data):
    allData = {}
    for dataset in Datasets:
        out = []
        dset = readFromDict(data, dataset, {})
        for algo in Algos:
            algoData = readFromDict(dset, algo, {})
            perf = [algo]
            for timing in Timings:
                perf.append(readFromDict(algoData, timing))
            metricData = readFromDict(algoData, "accuracy", {})
            for metric in Metrics:
                perf.append(readFromDict(metricData, metric))
            out.append(perf)
        allData[dataset] = out
    return allData

def writeCsv(allData):
    writer = csv.writer(sys.stdout)
    for dataset in Datasets:
        header = [dataset] + AllMetrics
        writer.writerow(header)
        for row in allData[dataset]:
            writer.writerow(row)
    return

def main():
    data = loadAllPerfData(sys.argv[1:])
    table = combinePerfData(data)
    writeCsv(table)
    return

if __name__ == '__main__':
    main()

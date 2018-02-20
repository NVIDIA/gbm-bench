#!/usr/bin/env python

import sys
import json
import string
import os
import csv

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

def getAllDatasets(data):
    return data.keys()

def getAllAlgos(data):
    algos = {}
    for dset in data.keys():
        for algo in data[dset].keys():
            algos[algo] = 1
    return algos.keys()

def readFromDict(hashmap, key, defVal="-na-"):
    d = hashmap[key] if key in hashmap else defVal
    return d

def combinePerfData(data, datasets, algos):
    allData = {}
    for dataset in datasets:
        out = []
        dset = readFromDict(data, dataset, {})
        for algo in algos:
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

def writeCsv(allData, datasets):
    writer = csv.writer(sys.stdout)
    for dataset in sorted(datasets):
        header = [dataset] + AllMetrics
        writer.writerow(header)
        for row in allData[dataset]:
            writer.writerow(row)
    return

def main():
    data = loadAllPerfData(sys.argv[1:])
    datasets = getAllDatasets(data)
    algos = getAllAlgos(data)
    table = combinePerfData(data, datasets, algos)
    writeCsv(table, datasets)
    return

if __name__ == '__main__':
    main()

#!/usr/bin/env python
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

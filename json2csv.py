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
import os
import csv

TIMINGS = ["train_time", "test_time"]
METRICS = ["AUC", "Accuracy", "F1", "Precision", "Recall", "MeanAbsError", "MeanSquaredError",
           "MedianAbsError"]
ALLMETRICS = TIMINGS + METRICS


def load_perf_data(json_file):
    file = open(json_file, "r")
    data = json.load(file)
    file.close()
    return data


def load_all_perf_data(files):
    data = {}
    for json_file in files:
        dataset = os.path.basename(json_file)
        dataset = dataset.replace(".json", "")
        data[dataset] = load_perf_data(json_file)
    return data


def get_all_datasets(data):
    return data.keys()


def get_all_algos(data):
    algos = {}
    for dset in data.keys():
        for algo in data[dset].keys():
            algos[algo] = 1
    return algos.keys()


def read_from_dict(hashmap, key, def_val="-na-"):
    return hashmap[key] if key in hashmap else def_val


def combine_perf_data(data, datasets, algos):
    all_data = {}
    for dataset in datasets:
        out = []
        dset = read_from_dict(data, dataset, {})
        for algo in algos:
            algo_data = read_from_dict(dset, algo, {})
            perf = [algo]
            for timing in TIMINGS:
                perf.append(read_from_dict(algo_data, timing))
            metric_data = read_from_dict(algo_data, "accuracy", {})
            for metric in METRICS:
                perf.append(read_from_dict(metric_data, metric))
            out.append(perf)
        all_data[dataset] = out
    return all_data


def write_csv(all_data, datasets):
    writer = csv.writer(sys.stdout)
    header = ['dataset', 'algorithm'] + ALLMETRICS
    writer.writerow(header)
    for dataset in sorted(datasets):
        for row in all_data[dataset]:
            writer.writerow([dataset] + row)


def main():
    data = load_perf_data(sys.argv[1])
    datasets = get_all_datasets(data)
    algos = get_all_algos(data)
    table = combine_perf_data(data, datasets, algos)
    write_csv(table, datasets)


if __name__ == '__main__':
    main()

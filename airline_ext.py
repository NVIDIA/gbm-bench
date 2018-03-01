
from __future__ import print_function

import airline
from airline import metrics, catMetrics, benchmarks


def prepareImpl(dbFolder, testSize, shuffle, nrows):
    rows = 165543376 if nrows is None else nrows
    return airline.prepareImplCommon(dbFolder, testSize, shuffle,
                                     "airline_full.csv", rows)

def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 0.2, True, nrows)

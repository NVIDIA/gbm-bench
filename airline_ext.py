
from __future__ import print_function

import airline
from airline import metrics, catMetrics, benchmarks


def prepareImpl(dbFolder, testSize, shuffle):
    return airline.prepareImplCommon(dbFolder, testSize, shuffle,
                                     "airline_full.csv", 165543376)

def prepare(dbFolder):
    return prepareImpl(dbFolder, 0.2, True)

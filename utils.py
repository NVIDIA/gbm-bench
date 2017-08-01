# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/utils.py
from __future__ import print_function
import os
import multiprocessing
import sys


def get_number_processors():
    try:
        num = os.cpu_count()
    except:
        num = multiprocessing.cpu_count()
    return num  


def print_sys_info():
    print("System  : %s" % sys.version)
    print("Xgboost : %s" % os.getenv("XG_COMMIT_ID"))
    print("LightGBM: %s" % os.getenv("LG_COMMIT_ID"))
    print("#jobs   : %d" % get_number_processors())

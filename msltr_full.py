# Source: https://www.microsoft.com/en-us/research/project/mslr/

import msltr
from utils import *


def prepare(dbFolder):
    return msltr.load_full(dbFolder, "MSLR-WEB30K.zip")


# NOTE: some benchmarks are disabled!
#  . xgb-gpu  out of memory
#  . cat-gpu  currently segfaults
benchmarks = {
    "xgb-cpu":      (True, XgbBenchmark, msltr.metrics,
                     dict(msltr.xgb_common_params, tree_method="exact",
                          nthread=msltr.nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, msltr.metrics,
                     dict(msltr.xgb_common_params, nthread=msltr.nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (False, XgbBenchmark, msltr.metrics,
                     dict(msltr.xgb_common_params, tree_method="gpu_exact")),
    "xgb-gpu-hist": (True, XgbBenchmark, msltr.metrics,
                     dict(msltr.xgb_common_params, tree_method="gpu_hist")),

    "lgbm-cpu":     (True, LgbBenchmark, msltr.metrics,
                     dict(msltr.lgb_common_params, nthread=msltr.nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, msltr.metrics,
                     dict(msltr.lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, msltr.metrics,
                     dict(msltr.cat_common_params, thread_count=msltr.nthreads)),
    "cat-gpu":      (False, CatBenchmark, msltr.metrics,
                     dict(msltr.cat_common_params, task_type="GPU")),
}

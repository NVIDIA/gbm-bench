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

import msltr
from utils import *


def prepare(dbFolder, nrows):
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

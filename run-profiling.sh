#!/bin/bash
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

OUT_DIR=../results/example
TMP_DIR=$OUT_DIR

# run xgb-gpu-hist for all datasets with different GPU counts
for bench in airline bosch football fraud higgs msltr msltr_full; do
  for ngpus in {1..2}; do
      TMP_FILE=$TMP_DIR/xgb-gpu-hist-$bench-$ngpus.csv
      OUT_FILE=$OUT_DIR/xgb-gpu-hist-$bench-$ngpus.csv
      EXTRA='{"debug_verbose":1}'
      # GPU objective function supported only for binary classification
      if [[ $bench != football -a $bench != msltr -a $bench != msltr_full ]]; then
          EXTRA='{"objective":"gpu:binary:logistic", "debug_verbose":1}'
      fi
      echo "benchmark $bench, ngpus=$ngpus"
			./runme.py -root ../datasets -dataset $bench -ngpus $ngpus -benchmarks xgb-gpu-hist \
                 -extra "$EXTRA"  2>&1 | \
          grep -E '^[A-Za-z0-9 ]+:\s+[0-9.]+s' | \
          sed 's/[0-9]s//' | sed 's/ Lifetime:/_Lifetime/' | sed 's/://' | tee $TMP_FILE
      if [[ "$TMP_DIR" != "$OUT_DIR" ]]; then
          cp "$TMP_FILE" "$OUT_FILE"
      fi
  done
done

#!/bin/bash

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

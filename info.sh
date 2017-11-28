#!/bin/bash

grep "model name" /proc/cpuinfo | head -n1
nvidia-smi -L | head -n1
uname -a
nvidia-smi | head -n3 | tail -n1
echo "XGBoost  Commit: $XG_COMMIT_ID"
echo "LightGBM Commit: $LG_COMMIT_ID"
echo "CatBoost Commit: $CAT_COMMIT_ID"

#!/bin/bash

grep "model name" /proc/cpuinfo | head -n1
nvidia-smi -L | head -n1
lsb_release -a | grep Description
nvidia-smi | head -n3 | tail -n1

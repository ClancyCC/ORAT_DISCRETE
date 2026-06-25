#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="./${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/qingcheng.zhu/hf_home"
export CUDA_VISIBLE_DEVICES="7"
export PATH="/usr/local/cuda-11.8/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WANDB_MODE="offline"

/home/estar/anaconda3/envs/orat/bin/python train_discrete/jpg_dataset.py \
  --mode export \
  --data-root-dir /data/qingcheng.zhu/TrackUAV/results/collect/trainset_1080_0519 \
  --output-pt-dir /data/qingcheng.zhu/orat0602/ORAT_DISCRETE-master/dataset_new \
  --expected-distance 5.0 \
  --collision-distance 2.0 \
  --max-episodes 0 \
  --batch-size 1 \
  --prompt drone

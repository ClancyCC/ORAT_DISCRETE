#!/bin/bash

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/qingcheng.zhu/hf_home"
export CUDA_VISIBLE_DEVICES="0"
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export WANDB_MODE="offline"

# 运行训练脚本
python train_offline_discrete.py \
    --buffer_path "/data/qingcheng.zhu/orat0602/ORAT_DISCRETE-master/dataset_new" \
    --run_name "discrete" \
    --episodes "5000" \
    --batch_size "8" \
    --lstm_seq_len "50" \
    --seed "1" \
    --save_every "250" \
    --use_wandb "0" \
    --use_tensorboard "1" \
    --save_dir "${WORKSPACE_FOLDER:-./}/trained_models_discrete"

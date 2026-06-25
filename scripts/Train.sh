#!/bin/bash
export HF_HOME="/data/qingcheng.zhu/hf_home"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="6"
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export WANDB_MODE="offline"


python train_offline.py \
    --buffer_path /data/qingcheng.zhu/Offline_RL_Active_Tracking/inperfect_expert_240px_v4_deva_mask_v1 \
    --use_wandb 0 \
    --use_tensorboard 1 \
    --log_dir ./logs \
    --save_dir ./trained_models \
    >> ./logs/train.log 2>&1

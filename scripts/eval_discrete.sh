#!/bin/bash

# 评估脚本：评估训练好的离散动作空间CQL-SAC模型

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/qingcheng.zhu/hf_home"
export CUDA_VISIBLE_DEVICES="0"
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export WANDB_MODE="offline"

# 运行评估脚本
python eval_offline_discrete.py \
    --run_name "eval_discrete" \
    --load_agent_model "${WORKSPACE_FOLDER:-./}/trained_models_discrete/CQL-SAC5000.pth" \
    --input_type "deva_cnn_lstm" \
    --eval_episodes "10" \
    --lstm_seq_len "50" \
    --hidden_size "256" \
    --learning_rate "3e-5" \
    --temperature "1.0" \
    --cql_weight "1.0" \
    --target_action_gap "10" \
    --with_lagrange "0" \
    --tau "5e-3" \
    --lstm_layer "1" \
    --lstm_out "64" \
    --seed "1" \
    --use_wandb "0" \
    --use_tensorboard "1" \
    --log_dir "${WORKSPACE_FOLDER:-./}/logs_eval" \
    --project_name "CQL-Eval"

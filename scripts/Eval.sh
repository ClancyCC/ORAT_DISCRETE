export PATHONPATH='./'
export HF_HOME="/data/qingcheng.zhu/hf_home"
export HF_ENDPOINT="https://hf-mirror.com"

python Eval_tracking_agent.py --env UnrealTrackGeneral-UrbanCity-ContinuousColor-v0 --chunk_size 1 --amp --min_mid_term_frames 5 --max_mid_term_frames 10 --detection_every 20 --prompt person.obstacles --load_agent_model /data/qingcheng.zhu/Offline_RL_Active_Tracking/wandb/offline-run-20260505_113907-h00wa7xw/files/trained_models/CQL-SAC-active-tracking_agentCQL-SAC1000.pth
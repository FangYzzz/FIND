#!/usr/bin/env bash

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-transport-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-transport-bc \
    --wandb_enable \
    --eval_env Transport \
    --rollout_freq 5000 \
    --steps 100000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 10000

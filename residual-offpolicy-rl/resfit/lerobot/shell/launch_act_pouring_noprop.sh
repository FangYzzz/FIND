#!/usr/bin/env bash

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-pouring \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmimicgen-pouring-bc \
    --wandb_enable \
    --eval_env TwoArmPouring \
    --rollout_freq 2000 \
    --disable_proprioceptive_obs \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 2000

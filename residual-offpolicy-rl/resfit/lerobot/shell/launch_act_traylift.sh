#!/usr/bin/env bash

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-lift-tray \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmimicgen-traylift-bc \
    --wandb_enable \
    --eval_env TwoArmLiftTray \
    --rollout_freq 5_000 \
    --steps 100_000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 10_000

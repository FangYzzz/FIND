#!/usr/bin/env bash

python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-drawer-cleanup \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmimicgen-drawercleanup-bc \
    --wandb_enable \
    --eval_env TwoArmDrawerCleanup \
    --rollout_freq 1000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 100 \
    --log_freq 100 \
    --save_freq 1000

#!/bin/bash
source /home/yuan/miniconda3/etc/profile.d/conda.sh
conda activate controller
pkill -9 gripper
launch_gripper.py gripper=franka_hand

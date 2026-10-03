#!/bin/bash
source /home/yuan/miniconda3/etc/profile.d/conda.sh
conda activate controller
pkill -9 run_server
launch_robot.py robot_client=franka_hardware

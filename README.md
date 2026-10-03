# FIND

Code for our paper, *Find Something You Can't Do: Agentic Real-World Reinforcement Learning for Self-Improving VLA Models*.

[Paper (arXiv)](https://arxiv.org/abs/2609.32069) | [Project webpage](https://fangyzzz.github.io/FIND.github.io/)

Residual reinforcement learning for fine-tuning vision-language-action policies with OpenPI.

## Acknowledgements

Many thanks to [amazon-far/residual-offpolicy-rl](https://github.com/amazon-far/residual-offpolicy-rl) for publicly releasing their code. This project builds on their residual off-policy reinforcement learning implementation.

## Getting started

Clone the repository together with its third-party submodules:

```bash
git clone --recurse-submodules https://github.com/FangYzzz/FIND.git self_vla
cd self_vla
```

For an existing checkout, run `git submodule update --init --recursive` from the repository root.

### 1. Install OpenPI

First, install [OpenPI](https://github.com/Physical-Intelligence/openpi) by following its [official installation instructions](https://github.com/Physical-Intelligence/openpi#installation).

For a fresh checkout, install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the following commands from the root of this repository:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git openpi
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..
```

Keep OpenPI in the `openpi/` directory at the repository root. This directory is intentionally ignored by Git and is not bundled with this project. If you already have an OpenPI checkout there, use the existing installation instead of cloning it again.

Refer to the official OpenPI documentation for system requirements, model checkpoints, and any updated installation steps.

### 2. Install Polymetis and Droid

Create a separate Conda environment named `controller` for the robot-side dependencies and install Polymetis by following the [official installation guide](https://facebookresearch.github.io/fairo/polymetis/installation.html). The guide's Conda installation uses Python 3.8:

```bash
conda create -n controller python=3.8
conda activate controller
conda install -c pytorch -c fair-robotics -c aihabitat -c conda-forge polymetis
```

Keep this environment active. From the root of this repository, install Droid, the OpenPI client, and `tyro`:

```bash
cd droid
pip install -e .
cd ../openpi/packages/openpi-client
pip install -e .
pip install tyro
cd ../../..
```

For a source build or a different robot firmware version, follow the corresponding instructions in the official Polymetis guide.

### 3. Install residual training dependencies

Use a separate **Python 3.11** environment for residual training, matching the environment used to prepare [requirements.txt](requirements.txt). Do not install these requirements into the Python 3.8 Polymetis environment or the OpenPI server environment.

From the repository root:

```bash
conda create -n residual python=3.11
conda activate residual
pip install -r requirements.txt
pip install -e ./residual-offpolicy-rl
pip install -e ./openpi/packages/openpi-client
export PYTHONPATH="$PWD/droid${PYTHONPATH:+:$PYTHONPATH}"
```

The `PYTHONPATH` setting exposes the Droid client code to training without installing Droid's legacy controller and GUI dependencies into this environment. Keep it set in the shell used to launch training. The full Droid/Polymetis installation remains in the environment from step 2.

The requirements cover the current `train_residual_td3_pi05_parallel_lang_chunk.py` workflow. They are a curated list, not a full environment freeze; pip resolves transitive dependencies automatically. GroundingDINO, OpenPI packages, local editable paths, and unrelated development/simulation tools are not listed. The OpenPI client is still required by training and is installed separately above. Shared libraries such as `transformers` and `torchvision` remain because the residual models use them.

NumPy is pinned to `1.26.4` to satisfy SciPy while staying below NumPy 2. Only `opencv-python-headless==4.11.0.86` is selected for training; do not add another OpenCV variant to this environment, as they share the same `cv2` namespace ([OpenCV installation guidance](https://pypi.org/project/opencv-python-headless/4.11.0.86/)). GUI camera-preview tools should use the robot-side environment.

Hardware and optional tools require separate setup:

- Online ZED camera access requires the ZED SDK and its matching `pyzed` wheel. They are not included in `requirements.txt`. The source environment's `pyzed 5.1` requires NumPy 2, which conflicts with the current OpenPI client's NumPy `<2` constraint; resolve this SDK/client compatibility issue before online rollouts.
- Video datasets decoded with TorchCodec require FFmpeg shared libraries; follow the [TorchCodec installation guide](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).


## Running

Start the policy server, robot server, and gripper server before launching training. Each long-running process needs its own terminal.

Replace `/path/to/self_vla`, checkpoint paths, and all `your_*_config` placeholders with your own settings. These configuration names must already be registered in OpenPI, Polymetis, or the residual trainer; the placeholders are not built-in configurations. Also configure your dataset paths, camera settings, robot/server addresses, and `OPENAI_API_KEY` before online training.

### 1. Start the OpenPI policy server

```bash
cd /path/to/self_vla/openpi
OPENPI_CONFIG="your_openpi_config"
CHECKPOINT_DIR="/absolute/path/to/checkpoint"

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
    --port=8008 \
    policy:checkpoint \
    --policy.config="$OPENPI_CONFIG" \
    --policy.dir="$CHECKPOINT_DIR"
```

Select a configuration in `openpi/src/openpi/training/config.py` that matches your checkpoint, observation format, and action representation. Port `8008` matches the residual client's `Args.pi05_port`; update both sides if you change it.

### 2. Start the robot and gripper servers

Before restarting the robot server, inspect any existing `run_server` processes. Only force-stop a verified stale server when the robot is idle; the following optional cleanup forcibly terminates all processes with that exact name:

```bash
pgrep -a -x run_server
sudo pkill -9 -x run_server
```

In the robot-server terminal:

```bash
conda activate controller
cd /path/to/self_vla
ROBOT_CLIENT_CONFIG="your_robot_client_config"

python droid/droid/fairo/polymetis/polymetis/python/scripts/launch_robot.py \
    robot_client="$ROBOT_CLIENT_CONFIG"
```

In a separate gripper-server terminal:

```bash
conda activate controller
cd /path/to/self_vla
GRIPPER_CONFIG="your_gripper_config"

python droid/droid/fairo/polymetis/polymetis/python/scripts/launch_gripper.py \
    gripper="$GRIPPER_CONFIG"
```

Choose the `robot_client` and `gripper` configurations that match your hardware from the Polymetis configuration groups.

### 3. Run residual training

In the training terminal:

```bash
conda activate residual
cd /path/to/self_vla
export PYTHONPATH="$PWD/droid${PYTHONPATH:+:$PYTHONPATH}"
cd residual-offpolicy-rl
RESIDUAL_CONFIG="your_residual_config"

python resfit/rl_finetuning/scripts/train_residual_td3_pi05_parallel_lang_chunk.py \
    --config-name="$RESIDUAL_CONFIG" \
    chunk_len=5 \
    algo.n_step=5 \
    algo.prefetch_batches=4 \
    algo.gamma=0.995 \
    algo.learning_starts=10000 \
    algo.critic_warmup_steps=10000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.003 \
    algo.stddev_min=0.003 \
    algo.stddev_schedule=0.003 \
    algo.random_action_noise_scale=0.05 \
    algo.buffer_size=100000 \
    agent.actor.action_scale=0.04 \
    agent.actor_lr=1e-6 \
    agent.clip_q_target_to_reward_range=true
```

Select or register your training configuration in `residual-offpolicy-rl/resfit/rl_finetuning/config/residual_td3.py`.

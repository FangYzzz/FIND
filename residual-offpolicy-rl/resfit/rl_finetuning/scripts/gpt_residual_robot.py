from __future__ import annotations

import contextlib
import dataclasses
import signal
from typing import Any, Optional
import copy
import cv2
import numpy as np
import requests
import torch
import json_numpy
from json_numpy import loads
from openpi_client import websocket_client_policy
import os
from datetime import datetime

from openpi_client import image_tools
from droid.franka_env import RobotEnv
json_numpy.patch()
from scipy.spatial.transform import Rotation as R
import time
from loguru import logger

from utils import process_policy_images, _extract_observation, prepare_image_256, to_hwc
from task_reward_generator import TaskRewardGenerator

DROID_CONTROL_FREQUENCY = 10

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt

class Args:
    # Fallback for prompts that are not listed in ``task_max_timesteps``.
    max_timesteps: int = 200

    # Maximum episode length for each task. Keep the keys identical to the
    # prompts produced by TaskRewardGenerator (or passed in during evaluation).
    task_max_timesteps: dict[str, int] = {
        "Put a cube into the bowl": 160,
        "Take the cube out of the bowl": 140,
        "Stack one cube on the other cube": 200,
        "Take the top cube off the other cube": 130,
        "Open the drawer": 140,
        "Close the drawer": 100,
        "Hang the mug on the mug tree": 210,
        "Take the mug off the mug tree": 340,
    }

    gpt_host: str = "127.0.0.1"
    gpt_port: int = 8007

    pi05_host: str = "127.0.0.1"
    pi05_port: int = 8008

    # DROID uses 0=open and 1=closed. Require consecutive predictions before
    # accepting a binary gripper state change.
    gripper_close_threshold: float = 0.9 # 0.994-1.004
    gripper_open_threshold: float = 0.1 # -0.004-0.003
    gripper_confirm_steps: int = 2


class BasePolicy:
    def __init__(
        self,
        main_host,
        main_port,
        action_scaler,
        state_standardizer,
        args=Args,
        connect_robot=True,
        save_images=True,
        candidate_tasks: Optional[list[str]] = None,
        success_rate_window_size: int = 20,
        min_task_sample_probability: float = 0.05,
        prioritize_low_success_tasks: bool = True,
    ):
        self.device = torch.device("cuda")
        self.training = False
        self.server = f"http://{main_host}:{main_port}"
        self.action_scaler = action_scaler
        self._last_base_action = None
        self.state_standardizer = state_standardizer
        self.save_images = bool(save_images)
        self.candidate_tasks = candidate_tasks
        self.success_rate_window_size = int(success_rate_window_size)
        self.min_task_sample_probability = float(min_task_sample_probability)
        self.prioritize_low_success_tasks = bool(prioritize_low_success_tasks)
        self.base_action_buffer =[]
        self.rl_token_encoder = None
        self.rl_token_obs_key = "observation.rl_token"
        self._last_rl_token = None
        # Keep the commanded gripper state across independently planned pi0
        # chunks.  The measured gripper position can sit near the middle of
        # its range while holding an object, so re-inferring open/closed from
        # that measurement at every 35-step boundary can cause a false open.
        self._gripper_command_state: Optional[int] = None
        self._gripper_close_count = 0
        self._gripper_open_count = 0
        # Environment step at which the online token was computed. This lets
        # ``step_chunk`` refresh the token exactly at residual decision
        # boundaries without duplicating a query when pi0 has just replanned
        # the base-action chunk at the same observation.
        self._last_rl_token_step = None

        self.env = None
        self.args = args
        self.pi05_client = websocket_client_policy.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
        self.text = "put the cube into the bowl"
        self.max_timesteps = args.max_timesteps
        self.t_step = 0
        self.round = 0
        self.reward = 0.0
        self.obs = None
        self.last_excution_time  = time.time()
        self.eefpose = None
        self.gripper_position = None
        self.task_reward_generator = None
        self.evaluation = False
        if connect_robot:
            self.connect_online_resources(required=True)
        else:
            print(
                "[BasePolicy] WARNING: robot connection skipped during offline "
                "embedding/VAE processing; dataset observations will be used."
            )

    def connect_robot(self, required=False):
        """Lazily connect hardware; offline VLA processing does not need it."""
        if self.env is not None:
            return True
        try:
            self.env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
            print("[BasePolicy] robot connected")
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                "[BasePolicy] WARNING: robot is not connected. Offline embedding/VAE "
                f"processing can continue without it. Details: {exc}"
            )
            if required:
                raise RuntimeError("Robot connection is required for online rollout") from exc
            return False

    def _stabilize_gripper_chunk(
        self, raw_gripper: np.ndarray, current_position
    ) -> np.ndarray:
        """Debounce commands while latching state across action chunks."""
        close_threshold = float(self.args.gripper_close_threshold)
        open_threshold = float(self.args.gripper_open_threshold)
        confirm_steps = int(self.args.gripper_confirm_steps)
        if not (0.0 <= open_threshold < close_threshold <= 1.0):
            raise ValueError(
                "Expected 0 <= gripper_open_threshold < "
                "gripper_close_threshold <= 1"
            )
        if confirm_steps < 1:
            raise ValueError("gripper_confirm_steps must be >= 1")
        raw = np.asarray(raw_gripper, dtype=np.float32).reshape(-1)
        if self._gripper_command_state is None:
            if current_position is None:
                raise ValueError(
                    "current gripper position is required to initialize stabilization"
                )
            current = float(np.asarray(current_position).reshape(-1)[0])
            state = int(current >= 0.5)
        else:
            state = self._gripper_command_state

        start_state = state
        stable = np.empty_like(raw)
        close_count = self._gripper_close_count
        open_count = self._gripper_open_count

        for index, value in enumerate(raw):
            if np.isfinite(value) and value >= close_threshold:
                close_count += 1
                open_count = 0
            elif np.isfinite(value) and value <= open_threshold:
                open_count += 1
                close_count = 0
            else:
                close_count = 0
                open_count = 0

            if state == 0 and close_count >= confirm_steps:
                state = 1
                close_count = 0
            elif state == 1 and open_count >= confirm_steps:
                state = 0
                open_count = 0
            stable[index] = state

        # A new pi0 plan is requested only after the previous plan is consumed.
        # Preserve the final planned command and any partial debounce sequence
        # so the next chunk begins from the same logical gripper state.
        self._gripper_command_state = state
        self._gripper_close_count = close_count
        self._gripper_open_count = open_count

        suppressed_close_spikes = int(
            np.count_nonzero((raw >= close_threshold) & (stable == 0))
        )
        if suppressed_close_spikes:
            print(
                f"[gripper-filter] suppressed {suppressed_close_spikes} "
                f"unconfirmed close prediction(s); raw range="
                f"[{raw.min():.3f}, {raw.max():.3f}]; "
                f"latched state {start_state}->{state}"
            )
        return stable

    def reset_gripper_filter_state(self, state: Optional[int] = None) -> None:
        """Reset the latch at a real or dataset episode boundary."""
        if state not in (None, 0, 1):
            raise ValueError(
                "gripper filter state must be None, 0 (open), or 1 (closed)"
            )
        self._gripper_command_state = state
        self._gripper_close_count = 0
        self._gripper_open_count = 0

    def connect_online_resources(self, required=False):
        """Connect resources needed only by real-robot rollout/evaluation."""
        try:
            if self.task_reward_generator is None:
                self.task_reward_generator = TaskRewardGenerator(
                    save_images=self.save_images,
                    candidate_tasks=self.candidate_tasks,
                    success_rate_window_size=self.success_rate_window_size,
                    min_task_sample_probability=self.min_task_sample_probability,
                    prioritize_low_success_tasks=self.prioritize_low_success_tasks,
                )
                print("[BasePolicy] reward camera/task generator connected")
            robot_connected = self.connect_robot(required=required)
            return self.task_reward_generator is not None and robot_connected
        except Exception as exc:  # noqa: BLE001
            print(
                "[BasePolicy] WARNING: online robot resources are unavailable. "
                f"Offline processing is unaffected. Details: {exc}"
            )
            if required:
                raise RuntimeError(
                    "Franka and the ZED reward camera are required for online rollout"
                ) from exc
            return False

    def _require_robot(self):
        if self.env is None or self.task_reward_generator is None:
            self.connect_online_resources(required=True)

    def get_task_max_timesteps(self, task_prompt: Optional[str] = None) -> int:
        """Return the configured episode length for the active task."""
        task = task_prompt if task_prompt is not None else self.text
        max_timesteps = self.args.task_max_timesteps.get(
            task, self.args.max_timesteps
        )
        if not isinstance(max_timesteps, int) or max_timesteps <= 0:
            raise ValueError(
                f"max_timesteps for task {task!r} must be a positive integer, "
                f"got {max_timesteps!r}"
            )
        return max_timesteps

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])
        if self._last_rl_token is not None:
            token = self._last_rl_token.to(base_naction.device)
            # Observations use a leading environment/batch dimension.  Keeping
            # the token consistent with state/base_action prevents replay
            # insertion from mistaking token_dim for num_envs and selecting a
            # single scalar with ``token[i]``.
            if token.ndim == 1:
                token = token.unsqueeze(0)
            augmented_obs[self.rl_token_obs_key] = token

        return augmented_obs

    def set_rl_token_encoder(self, encoder, obs_key="observation.rl_token"):
        self.rl_token_encoder = encoder
        self.rl_token_obs_key = obs_key

    def _set_rl_token_from_result(self, result):
        if "vla_embedding" not in result or "vla_embedding_mask" not in result:
            raise RuntimeError(
                f"pi0 response is missing RL-token features: {sorted(result.keys())}"
            )
        emb = torch.from_numpy(
            np.array(result["vla_embedding"], dtype=np.float32, copy=True)
        ).to(self.device).unsqueeze(0)
        valid = torch.from_numpy(
            np.array(result["vla_embedding_mask"], dtype=np.bool_, copy=True)
        ).to(self.device).unsqueeze(0)
        self._last_rl_token = self.rl_token_encoder.encode(emb, ~valid).squeeze(0)
        self._last_rl_token_step = self.t_step

    @torch.no_grad()
    def refresh_online_rl_token(self, task_prompt=None):
        """Refresh the current-state token without replanning base actions."""
        if self.rl_token_encoder is None:
            return False
        if self._last_rl_token_step == self.t_step:
            return False

        prompt = self.text if task_prompt is None else task_prompt
        request_data = self.get_obs_for_base(prompt)
        request_data["return_vla_embedding"] = True
        request_data["embedding_only"] = True
        result = self.pi05_client.infer(request_data)
        self._set_rl_token_from_result(result)
        return True

    @torch.no_grad()
    def get_offline_vla_embedding(self, raw_obs, task_prompt):
        """Request final pi0 camera+language features without using a text side encoder."""
        obs_right = raw_obs["wrist_image_left"].detach().cpu().numpy()
        obs_wrist = raw_obs["exterior_image_2_left"].detach().cpu().numpy()
        eef_pose = raw_obs["eef_position"].detach().cpu().numpy()
        gripper = raw_obs["gripper_position"].detach().cpu().numpy()
        _, right, wrist = process_policy_images(obs_left=None, obs_right=obs_right, obs_wrist=obs_wrist)
        if eef_pose.ndim == 2:
            eef_pose = eef_pose.squeeze(0)
        result = self.pi05_client.infer({
            "observation/wrist_image_left": image_tools.resize_with_pad(right, 224, 224),
            "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist, 224, 224),
            "observation/eef_position": eef_pose,
            "observation/gripper_position": gripper,
            "prompt": task_prompt,
            "return_vla_embedding": True,
            "embedding_only": True,
        })
        if "vla_embedding" not in result or "vla_embedding_mask" not in result:
            raise RuntimeError(
                "The pi0 server did not return RL-token features. "
                f"Response keys were {sorted(result.keys())}. Restart the server from "
                "/home/yuan/self_vla/openpi after applying the RL-token policy changes."
            )
        # The msgpack decoder may expose read-only NumPy views. Copy before
        # converting so PyTorch never receives non-writable storage.
        return (torch.from_numpy(np.array(result["vla_embedding"], dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(result["vla_embedding_mask"], dtype=np.bool_, copy=True)))

    def update_obs(self):
        self._require_robot()
        self.obs = _extract_observation(
            self.env.get_observation(),
            save_to_disk=False,
        )
        self.gripper_position = self.obs["gripper_position"]

    def save_debug_images(self, obs_left=None, obs_right=None, obs_wrist=None, save_dir="debug_images"):
        if not self.save_images:
            return
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        images = {
            "right": obs_right,
            "wrist": obs_wrist,
        }

        for name, img in images.items():
            path = os.path.join(save_dir, f"{timestamp}_{name}.png")

            if img.ndim == 3 and img.shape[-1] == 3:
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_to_save = img
            cv2.imwrite(path, img_to_save)

    def get_obs_for_base(self, task_prompt):
        obs_right = copy.deepcopy(self.obs["right_image"])
        obs_wrist = copy.deepcopy(self.obs["wrist_image"])
        eef_pose = copy.deepcopy(self.obs["cartesian_position"]) # [x, y, z, roll, pitch, yaw]
        gripper_position = copy.deepcopy(self.obs["gripper_position"])

        self.eefpose = copy.deepcopy(eef_pose)

        eef_pos = eef_pose[:3]
        eef_rpy = eef_pose[3:6]  # [x, y, z, roll, pitch, yaw]
        eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
        if eef_quat[3] < 0:  # Canonicalize the quaternion sign to avoid discontinuities.
            eef_quat = -eef_quat
        eef_pose_quat = np.concatenate([eef_pos, eef_quat], axis=-1)

        obs_right = prepare_image_256(obs_right)
        obs_wrist = prepare_image_256(obs_wrist)
        self.save_debug_images(obs_right = obs_right, obs_wrist = obs_wrist)

        return{
            "observation/wrist_image_left": image_tools.resize_with_pad(obs_right, 224, 224),
            "observation/exterior_image_2_left": image_tools.resize_with_pad(obs_wrist, 224, 224),
            "observation/eef_position": eef_pose_quat,
            "observation/gripper_position": gripper_position,
            "prompt": task_prompt,  # instruction
        }

    def get_obs_for_residual(self, base_naction):
        obs_right = torch.as_tensor(prepare_image_256(to_hwc(copy.deepcopy(self.obs["right_image"]))), dtype=torch.uint8, device=self.device)
        obs_wrist = torch.as_tensor(prepare_image_256(to_hwc(copy.deepcopy(self.obs["wrist_image"]))), dtype=torch.uint8, device=self.device)
        eef_pose = copy.deepcopy(np.asarray(self.obs["cartesian_position"], dtype=np.float32))
        gripper_position = self.obs["gripper_position"]

        eef_pos = eef_pose[:3]
        eef_rpy = eef_pose[3:6]
        eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()  # [qx, qy, qz, qw]
        if eef_quat[3] < 0:  # Canonicalize the quaternion sign to avoid discontinuities.
            eef_quat = -eef_quat
        eef_pose_quat = np.concatenate([eef_pos, eef_quat], axis=-1)

        eef_pose_quat = torch.as_tensor(eef_pose_quat, dtype=torch.float32, device=self.device)
        gripper_position = torch.as_tensor(gripper_position, dtype=torch.float32, device=self.device)
        base_naction = torch.as_tensor(base_naction, dtype=torch.float32, device=self.device)

        if obs_right.ndim == 3:
            obs_right = obs_right.unsqueeze(0)
        if obs_wrist.ndim == 3:
            obs_wrist = obs_wrist.unsqueeze(0)

        if eef_pose_quat.ndim == 1:
            eef_pose_quat = eef_pose_quat.unsqueeze(0)
        if gripper_position.ndim == 1:
            gripper_position = gripper_position.unsqueeze(0)
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)

        state = torch.cat([eef_pose_quat, gripper_position], dim=-1).to(self.device)

        obs = {
            "observation.state": state,
            "observation.base_action": base_naction,
            "observation.images.exterior_image_2_left": obs_right,
            "observation.images.wrist_image_left": obs_wrist,
            "text": self.text,
        }
        augemnted_obs = self._augment_obs(obs, base_naction)

        return augemnted_obs

    def reset(self, task_prompt=None, evaluate_previous=True):
        self._require_robot()
        # A base-action plan must never cross a task/episode boundary. Clear it
        # before any environment or VLA operation so an exception during reset
        # cannot leave stale actions available to the next task.
        stale_actions = len(self.base_action_buffer)
        self.base_action_buffer.clear()
        self._last_base_action = None
        self._last_rl_token = None
        self._last_rl_token_step = None
        print(f"[BasePolicy.reset] cleared {stale_actions} buffered base action(s)")
        self.t_step = 0
        self.env.reset()  # 1.593s
        # RobotEnv.reset() opens the gripper.  Seed the command latch from that
        # known reset state instead of reclassifying a later grasp aperture.
        self.reset_gripper_filter_state(state=0)

        self.pi05_client.reset()
        print("robot reset successfully")

        if self.round != 0 and evaluate_previous:
            self.update_obs()
            self.reward = float(
                self.task_reward_generator.reward_generation(
                    self.obs["right_image"],
                    task_prompt,
                    # Training keeps the separator immediately after its
                    # reward block. Evaluation prints it after the per-episode
                    # result and accumulated progress instead.
                    log_separator=not self.evaluation,
                )
            )  # 0 or 1
            time.sleep(7)

        print(f"---------------------------- trajectory {self.round} ----------------------------")
        self.update_obs()
        if self.evaluation ==False:
            self.text, self.round = self.task_reward_generator.task_generation(
                self.obs["right_image"]
            )
        elif task_prompt is not None:
            self.task_reward_generator.start_task(self.obs["right_image"])
            self.round = self.task_reward_generator.round
            self.text = task_prompt
        self.max_timesteps = self.get_task_max_timesteps(task_prompt)
        print("current task:", self.text)
        print("max timesteps:", self.max_timesteps)
        if task_prompt==None:
            obs_for_pi0 = self.get_obs_for_base(self.text)
        else:
            obs_for_pi0 = self.get_obs_for_base(task_prompt)
        action_base = self.get_online_action_base(obs_for_pi0)

        self.base_action_buffer.clear()
        for action_ in action_base:
            self.base_action_buffer.append(action_)
        action_base_ = self.base_action_buffer.pop(0)
        base_naction = self.action_scaler.scale(action_base_)
        self._last_base_action = base_naction
        if base_naction.ndim == 1:
            base_naction = base_naction.unsqueeze(0)

        obs = self.get_obs_for_residual(base_naction)

        if task_prompt == None:
            return obs, self.text
        return obs

    def get_offline_action_base(self, raw_obs, task_prompt):
        if len(self.base_action_buffer)==0:
            query_action_base = True
        else:
            query_action_base = False

        obs_right = raw_obs["wrist_image_left"].detach().cpu().numpy()
        obs_wrist = raw_obs["exterior_image_2_left"].detach().cpu().numpy()
        eef_pose = raw_obs["eef_position"].detach().cpu().numpy()  # [x, y, z, qx, qy, qz, qw]
        gripper_position = raw_obs["gripper_position"].detach().cpu().numpy()

        left_resized, right_resized, wrist_resized = process_policy_images(obs_left=None, obs_right = obs_right, obs_wrist = obs_wrist)


        if query_action_base:
            if eef_pose.ndim == 2 :
                eef_pose = eef_pose.squeeze(0)

            request_data = {
                "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
                "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
                "observation/eef_position": eef_pose,
                "observation/gripper_position": gripper_position,
                "prompt": task_prompt,  # instruction
            }

            if self.rl_token_encoder is not None:
                request_data["return_vla_embedding"] = True
            result = self.pi05_client.infer(request_data)
            pred_action_chunk = result["actions"]
            if self.rl_token_encoder is not None:
                self._set_rl_token_from_result(result)
            assert pred_action_chunk.shape == (50, 8) # 10,8
            action = pred_action_chunk[:35]

            action = np.asarray(action).copy()
            action[:, -1] = self._stabilize_gripper_chunk(
                action[:, -1], gripper_position
            )

            # Convert position deltas to absolute targets.
            action[:,:3] = action[:,:3] + eef_pose[:3]
            action_chunk = np.asarray(action, dtype=np.float32) 
        else:
            action_chunk = None

        if query_action_base:
            self.base_action_buffer.clear()
            for action in action_chunk:
                self.base_action_buffer.append(action)
        action_base = self.base_action_buffer.pop(0)

        return action_base

    def get_online_action_base(self, obs_for_pi0):
        if self.rl_token_encoder is not None:
            obs_for_pi0["return_vla_embedding"] = True
        result = self.pi05_client.infer(obs_for_pi0)
        pred_action_chunk = result["actions"]
        if self.rl_token_encoder is not None:
            self._set_rl_token_from_result(result)
        assert pred_action_chunk.shape == (50, 8) # 10,8
        action = pred_action_chunk[:35]

        action = np.asarray(action).copy()
        action[:, -1] = self._stabilize_gripper_chunk(
            action[:, -1], self.gripper_position
        )
        # Convert position deltas to absolute targets.
        print(f"self.eefpose[:3] {self.eefpose[:3]}")
        action[:,:3] = action[:,:3] + self.eefpose[:3]
        action = np.asarray(action, dtype=np.float32) 
        action = torch.as_tensor(action, device=self.device)

        return action

    def step(self, residual_action, task_prompt = None, evaluation=False):
        # For safe position-only residual learning, leave orientation and
        # gripper entirely under the base policy's control. Clone first so
        # logging/training callers do not observe an unexpected in-place edit.
        base_action = torch.as_tensor(
            self._last_base_action, dtype=torch.float32, device=self.device
        ).clone()
        residual_action = residual_action.clone()
        residual_action[..., 3:7] = 0.0
        residual_action[..., 7] = 0.0
        combined_action = base_action + residual_action
        unscaled_combined_action = self.action_scaler.unscale(combined_action)
        self.evaluation = evaluation
        if len(self.base_action_buffer)<1:
            query_action_base = True
        else:
            query_action_base = False


        next_action_chunk, done = self.get_transition(combined_action=unscaled_combined_action, query_action_base = query_action_base,task_prompt = task_prompt) # TODO: terminated, truncated?


        if done:
            if task_prompt ==None:
                next_obs, task_prompt = self.reset()
            else:
                next_obs = self.reset(task_prompt=task_prompt)

            reward = self.reward
        else:
            reward = 0.0
            if query_action_base:
                self.base_action_buffer.clear()
                for action_ in next_action_chunk:
                    self.base_action_buffer.append(action_)

            next_base_action_ = self.base_action_buffer.pop(0)
            base_naction = self.action_scaler.scale(next_base_action_)
            self._last_base_action = base_naction
            if base_naction.ndim == 1:
                base_naction = base_naction.unsqueeze(0)
            next_obs = self.get_obs_for_residual(base_naction)


        info = {}
        info["base_action"] = base_action
        info["scaled_action"] = combined_action
        info["combined_action"] = unscaled_combined_action
        info["residual_action"] = residual_action
        info["task_prompt"] = self.text

        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([done], dtype=torch.bool, device=self.device)

        return next_obs, reward, done, info

    def current_base_chunk(self, chunk_len):
        """Flat SCALED base-action chunk (1, chunk_len*dim) at the current step.
        ``_last_base_action`` is already scaled; ``base_action_buffer`` holds the raw
        upcoming steps of the current base plan (scaled here). Pads by repeating the
        last valid entry when the base plan runs out.
        """
        first = torch.as_tensor(self._last_base_action, dtype=torch.float32, device=self.device).reshape(-1)
        parts = [first]
        for j in range(chunk_len - 1):
            if j < len(self.base_action_buffer):
                raw = torch.as_tensor(self.base_action_buffer[j], dtype=torch.float32, device=self.device)
                parts.append(self.action_scaler.scale(raw).reshape(-1))
            else:
                parts.append(parts[-1].clone())
        return torch.cat(parts, dim=-1).unsqueeze(0)
    def step_chunk(self, residual_flat, task_prompt=None, evaluation=False):
        """Open-loop execute one residual chunk.
        ``residual_flat``: (1, chunk_len*dim). ``step`` already adds its own per-step
        base action, so we only feed the residual slice each step. Returns the next
        chunk-start obs, the executed combined action chunk (1, chunk_len*dim), the
        accumulated (undiscounted) reward, done, and info. Reward is left undiscounted
        because rewards are sparse/terminal (0/1) and eval checks ``reward > 0.9``;
        cross-chunk discounting is handled by ``gamma_chunk = gamma**H`` in the buffer.
        """
        per_step_dim = torch.as_tensor(self._last_base_action).reshape(-1).shape[0]
        residual = residual_flat.reshape(-1, per_step_dim)  # (H, m)
        H = residual.shape[0]
        base_parts = []
        combined_parts = []
        executed_residual_parts = []
        reward_sum = 0.0
        done = False
        info = {}
        next_obs = None
        for h in range(H):
            next_obs, r, d, info = self.step(
                residual_action=residual[h : h + 1], task_prompt=task_prompt, evaluation=evaluation
            )
            base_parts.append(info["base_action"].reshape(-1))
            combined_parts.append(info["scaled_action"].reshape(-1))  # (m,) scaled
            executed_residual_parts.append(info["residual_action"].reshape(-1))
            reward_sum += float(r.sum().item())
            if bool(d.any()):
                done = True
                break
        executed_steps = len(combined_parts)
        executed_base = torch.stack(base_parts, dim=0)
        executed_combined = torch.stack(combined_parts, dim=0)
        executed_residual = torch.stack(executed_residual_parts, dim=0)
        while len(combined_parts) < H:  # pad on early termination
            combined_parts.append(combined_parts[-1].clone())

        # The actor chooses one residual chunk from ``next_obs``. Refresh the
        # VLA token once here so its cadence matches residual decisions rather
        # than the independently cached 35-step pi0 base-action plan. If the
        # base policy was replanned on the final environment step, the step
        # marker makes this a no-op because that inference already returned a
        # token for the same observation.
        if not done and self.rl_token_encoder is not None:
            self.refresh_online_rl_token(task_prompt=task_prompt)
            token = self._last_rl_token.to(self.device)
            if token.ndim == 1:
                token = token.unsqueeze(0)
            next_obs[self.rl_token_obs_key] = token

        combined = torch.stack(combined_parts, dim=0)  # (H, m) scaled
        combined_flat = combined.reshape(1, -1)  # (1, H*m) -> buffer action
        reward = torch.as_tensor([reward_sum], dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor([done], dtype=torch.bool, device=self.device)
        out_info = {
            "scaled_action": combined_flat,  # chunk-level combined action (for buffer)
            "combined_action": self.action_scaler.unscale(combined),  # (H, m) unscaled (for eval plots)
            "residual_action": residual,  # (H, m)
            # Exact, unpadded actions for per-episode oscillation diagnostics.
            "executed_steps": executed_steps,
            "executed_base_action": self.action_scaler.unscale(executed_base),
            "executed_combined_action": self.action_scaler.unscale(executed_combined),
            "executed_residual_action": executed_residual,
            "task_prompt": self.text,
        }
        return next_obs, combined_flat, reward, done_t, out_info

    def get_transition(self, combined_action, query_action_base, task_prompt):
        if isinstance(combined_action, torch.Tensor):
            combined_action = combined_action.detach().cpu().numpy()  # type: torch.Tensor
        combined_action = np.asarray(combined_action, dtype=np.float32)

        if combined_action.ndim == 2 and combined_action.shape[0] == 1:
            combined_action = combined_action[0]  # shape: torch.Size([1, 8]) -> (8,)
        assert combined_action.ndim == 1, f"Expected 1D combined_action, got shape {combined_action.shape}"

        pos = combined_action[:3]
        q_action = combined_action[3:7]
        gripper = combined_action[7:]

        # Enforce the end-effector floor limit.
        z_min = 0.20
        pos[2] = max(pos[2], z_min)

        norm = np.linalg.norm(q_action, keepdims=True)
        q_action = q_action / np.clip(norm, 1e-12, None)
        sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
        q_action = q_action * sign


        rpy_cmd = R.from_quat(q_action).as_euler('xyz', degrees=False)
        combined_action = np.concatenate([pos, rpy_cmd, gripper], axis=-1)
        elapsed_time = time.time() - self.last_excution_time
        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        self.env.step(combined_action)
        self.last_excution_time = time.time()
        self.t_step += 1
        self.update_obs() # update new obs
        print("trajectory steps:::::::::::::::::::::::", self.t_step)

        terminated = False
        # Each task can have its own time limit. Since t_step is incremented
        # after executing the action, >= max_timesteps permits exactly that
        # many environment steps.
        truncated = self.t_step >= self.max_timesteps
        done = terminated | truncated

        if done:
            combined_action[2] +=0.1
            self.env.step(combined_action)
            done = done
            next_action_chunk = None
        else:
            if query_action_base:
                if task_prompt ==None:
                    obs_for_pi0 = self.get_obs_for_base(self.text)
                else:
                    obs_for_pi0 = self.get_obs_for_base(task_prompt)
                next_action_chunk = self.get_online_action_base(obs_for_pi0)
            else:
                next_action_chunk = None

        return next_action_chunk, done # , reward

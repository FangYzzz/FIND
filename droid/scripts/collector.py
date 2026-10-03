import contextlib
import dataclasses
import datetime
import faulthandler
import io
import json
import signal
import threading
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import json_numpy
import numpy as np
import requests
import torch
import tyro
from PIL import Image
from droid.robot_env import RobotEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
from fastapi import FastAPI

import copy
from torch import nn
from fastapi.responses import JSONResponse
import uvicorn

from resfit.rl_finetuning.off_policy.networks.encoder import VitEncoder, SiglipEncoder
from resfit.rl_finetuning.off_policy.rl.actor import Actor
from resfit.rl_finetuning.config.residual_td3 import ResidualTD3FrankaTomatoConfig
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer

json_numpy.patch()
faulthandler.enable()


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))
# Args
@dataclasses.dataclass
class Args:
    max_timesteps: int = 150

    # GPT server
    gpt_host: str = "127.0.0.1"
    gpt_port: int = 8007

    # PI05 server
    pi05_host: str = "127.0.0.1"
    pi05_port: int = 8008

    # Learner server: collector -> learner 
    learner_host: str = "127.0.0.1"
    learner_port: int = 8009

    # Saving
    trajectory_save_dir: str = "./collector_trajectories"
    push_trajectories_to_learner: bool = True

    # Collector loop
    control_hz: float = 15  # 0 means no rate limiting
    policy_poll_interval_steps: int = 10
    learner_request_timeout_s: float = 10.0

    # Residual policy
    action_dim: int = 8
    residual_scale: float = 1.0
    device: str = "cuda"
    actor_sync_strict: bool = True

    # Safety / action combine
    z_min: float = 0.225
    use_base_chunk_first_action: bool = True
    num_envs = 1


# Utils
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

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


def save_np_image(img: np.ndarray, path: str):
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    Image.fromarray(img).save(path)


def to_hwc_uint8(img) -> np.ndarray:
    """
    Convert image to HWC uint8 RGB.
    Supports:
    - [H, W, C]
    - [1, H, W, C]
    - [C, H, W]
    - [1, C, H, W]
    """
    if img is None:
        raise ValueError("img is None")

    img = np.asarray(img)

    # [1, ...] -> [...]
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape={img.shape}")
        img = img[0]

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim != 3:
        raise ValueError(f"Invalid image ndim={img.ndim}, shape={img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    return img


def hwc_to_chw_uint8(img_hwc: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 image to CHW uint8 image."""
    if img_hwc.ndim != 3 or img_hwc.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={img_hwc.shape}")
    return np.transpose(img_hwc, (2, 0, 1)).astype(np.uint8, copy=False)


def center_crop_square(img_hwc: np.ndarray) -> np.ndarray:
    """Center-crop HWC image to square."""
    h, w = img_hwc.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_hwc[y0:y0 + side, x0:x0 + side]


def resize_hwc(img_hwc: np.ndarray, size: int) -> np.ndarray:
    """Resize HWC uint8 image to (size, size)."""
    out = cv2.resize(img_hwc, (size, size), interpolation=cv2.INTER_AREA)
    return out.astype(np.uint8, copy=False)


def preprocess_camera_image(img, *, crop_square: bool = True) -> np.ndarray:
    """
    Standard preprocessing for raw camera image:
    - convert to HWC uint8
    - optional center crop to square
    """
    img_hwc = to_hwc_uint8(img)
    if crop_square:
        img_hwc = center_crop_square(img_hwc)
    return img_hwc


def process_policy_images(
    obs_left,
    obs_right,
    obs_wrist,
    *,
    base_size: int = 224,
    residual_size: int = 84,
) -> dict:
    """
    Unified image preprocessing for both:
    - base policy / pi05 input: 224x224 (with pad)
    - residual actor / learner input: residual_size x residual_size

    Returns:
    {
        "base": {
            "left": ...,
            "right": ...,
            "wrist": ...,
        },
        "residual": {
            "left": ...,
            "right": ...,
            "wrist": ...,
        }
    }
    """
    left = preprocess_camera_image(obs_left)
    right = preprocess_camera_image(obs_right)
    wrist = preprocess_camera_image(obs_wrist)

    # base policy branch: keep your original resize_with_pad behavior
    left_base = image_tools.resize_with_pad(left, base_size, base_size)
    right_base = image_tools.resize_with_pad(right, base_size, base_size)
    wrist_base = image_tools.resize_with_pad(wrist, base_size, base_size)

    # residual / learner branch: direct square resize to training size
    left_residual = resize_hwc(left, residual_size)
    right_residual = resize_hwc(right, residual_size)
    wrist_residual = resize_hwc(wrist, residual_size)

    return {
        "base": {
            "left": left_base,
            "right": right_base,
            "wrist": wrist_base,
        },
        "residual": {
            "left": left_residual,
            "right": right_residual,
            "wrist": wrist_residual,
        },
    }

def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        if "left_cam" in key:
            left_image = image_observations[key]
        elif "right_cam" in key:
            right_image = image_observations[key]
        elif "wrist_cam" in key:
            wrist_image = image_observations[key]

    # ZEDXCamera.read_camera() already converts the camera's BGRA frames to RGB.
    # Only drop the alpha channel here; reversing the channels again would turn
    # red objects blue before the image is sent to GPT.
    left_image = np.ascontiguousarray(left_image[..., :3])
    right_image = np.ascontiguousarray(right_image[..., :3])
    wrist_image = np.ascontiguousarray(wrist_image[..., :3])

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def rpy_pose_to_quat_pose(eef_pose_rpy: np.ndarray) -> np.ndarray:
    eef_pose_rpy = np.asarray(eef_pose_rpy, dtype=np.float32)
    pos = eef_pose_rpy[:3]
    rpy = eef_pose_rpy[3:6]
    quat = R.from_euler("xyz", rpy, degrees=False).as_quat()
    if quat[3] < 0:
        quat = -quat
    return np.concatenate([pos, quat], axis=-1).astype(np.float32)


def quat_action_to_rpy_action(combined_action_quat: np.ndarray, z_min: float) -> np.ndarray:
    combined_action_quat = np.asarray(combined_action_quat, dtype=np.float32).copy()

    pos = combined_action_quat[:3].copy()
    q_action = combined_action_quat[3:7].copy()
    gripper = combined_action_quat[7:].copy()

    pos[2] = max(float(pos[2]), float(z_min))

    norm = np.linalg.norm(q_action, keepdims=True)
    q_action = q_action / np.clip(norm, 1e-12, None)
    if q_action[3] < 0:
        q_action = -q_action

    rpy_cmd = R.from_quat(q_action).as_euler("xyz", degrees=False)
    env_action = np.concatenate([pos, rpy_cmd, gripper], axis=-1).astype(np.float32)
    return env_action


# Learner communication
class LearnerClient:
    def __init__(self, host: str, port: int, timeout_s: float = 10.0):
        self.server = f"http://{host}:{port}"
        self.timeout_s = timeout_s

    def push_transition(self, trajectory_payload: dict):
        resp = requests.post(
            f"{self.server}/push_transition",
            json=trajectory_payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def push_warmup_trajectory(self, trajectory_payload: dict):
        resp = requests.post(
            f"{self.server}/push_warmup_transition",
            json=trajectory_payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def get_latest_policy(self, current_version: int):
        resp = requests.get(
            f"{self.server}/get_latest_policy",
            params={"version": int(current_version)},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()


# Residual actor wrapper
class ResidualActorPolicy(nn.Module):
    """
    Collector-side inference-only residual policy.

    Contains:
    - encoders
    - actor

    Does NOT contain:
    - critic
    - target networks
    - optimizers
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        prop_shape: tuple[int],
        action_dim: int,
        rl_cameras: list[str] | str,
        cfg,
    ):
        super().__init__()

        if isinstance(rl_cameras, str):
            rl_cameras = [rl_cameras]
        assert len(rl_cameras) > 0
        assert len(prop_shape) == 1

        self.rl_cameras = rl_cameras
        self.cfg = cfg

        self.encoders = self._build_encoders(obs_shape)

        sample_encoder = self.encoders[0]
        repr_dim_single = int(sample_encoder.repr_dim)
        patch_repr_dim = int(sample_encoder.patch_repr_dim)
        repr_dim = repr_dim_single * len(self.rl_cameras)

        prop_dim = prop_shape[0] if cfg.use_prop else 0

        self.actor = Actor(
            repr_dim=repr_dim,
            patch_repr_dim=patch_repr_dim,
            prop_dim=prop_dim,
            action_dim=action_dim,
            cfg=cfg.actor,
            residual_actor=True,
        )

        self.version = 0

        self.to(cfg.device)
        self.eval()

    def load_policy_payload(self, payload: dict, strict: bool = True):
        self.encoders.load_state_dict(payload["encoders"], strict=strict)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.eval()

    def _build_encoders(self, obs_shape):
        encoders = nn.ModuleList()
        for _ in self.rl_cameras:
            if self.cfg.enc_type == "vit":
                enc = VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
            elif self.cfg.enc_type == "siglip":
                enc = SiglipEncoder(obs_shape, self.cfg.siglip).to(self.cfg.device)
            else:
                raise AssertionError(f"Unknown encoder type {self.cfg.enc_type}.")
            encoders.append(enc)
        return encoders

    @torch.no_grad()
    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for cam_idx, cam_name in enumerate(self.rl_cameras):
            data = obs[cam_name]

            if data.dtype == torch.uint8:
                data = data.float().div_(255.0)
            else:
                data = data.float()

            feat_cam = self.encoders[cam_idx].forward(data, flatten=False)
            feats.append(feat_cam)

        feat_all = torch.cat(feats, dim=1)
        return feat_all

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if obs[self.rl_cameras[0]].dim() == 3:
            should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], eval_mode: bool = True) -> torch.Tensor:
        """
        Input obs format:
        {
            camera keys: [H,W,C] or [1,H,W,C] or [3,H,W] etc,
            "observation.state": [D] or [1,D],
            "observation.base_action": ...
        }

        Output:
            residual action tensor, shape [action_dim] or [B, action_dim]
        """
        self.eval()

        obs = copy.copy(obs)
        unsqueezed = self._maybe_unsqueeze_(obs)

        feat = self._encode(obs)
        obs["feat"] = feat

        dist = self.actor.forward(obs, stddev=0.0)
        action = dist.mean if eval_mode else dist.sample()

        if unsqueezed:
            action = action.squeeze(0)

        return action


def decode_policy_payload_from_base64(b64_string: str) -> dict:
    raw = base64.b64decode(b64_string.encode("utf-8"))
    buffer = io.BytesIO(raw)
    payload = torch.load(buffer, map_location="cpu", weights_only=False)
    return payload


# Collector
class Collector:
    def __init__(
        self,
        env,
        args: Args,
        gpt_server: str,
        pi05_client,
        learner_client: LearnerClient,
        actor_policy: ResidualActorPolicy,
    ):
        self.env = env
        self.args = args
        self.gpt_server = gpt_server
        self.pi05_client = pi05_client
        self.learner_client = learner_client
        self.actor_policy = actor_policy

        self.text = "pick up the cube and place it into the bowl"
        self.t_step = 0
        self.round = 0
        self.reward = 0.0

        self.current_obs_packet: Optional[dict] = None
        self.current_trajectory: List[dict] = []
        self.total_transition_count = 0
        self.total_trajectory_count = 0
        self.policy_version = 0

        self.trajectory_save_dir = Path(args.trajectory_save_dir)
        self.trajectory_save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.state_standardizer = StateStandardizer(state_mean= None, state_std= None)
        self.action_scaler = ActionScaler(action_min=None, action_max=None)
        self.online_warmup = False
        self.online_move = False

    def run(self, port = 8010, host = "127.0.0.1"):
        self.app = FastAPI()
        self.app.post("/query_offline_action_base")(self.get_offline_action_base)
        self.app.post("/start_move")(self.start_move)
        uvicorn.run(self.app, host=host, port=port)

    def start_move(self, payload: Dict[Any, Any]):
        self.online_warmup = payload["online_warmup"]
        self.online_move = payload["online_move"]

    # Task & Reward generation
    def task_generation(self, obs_right):
        resp = requests.post(
            f"{self.gpt_server}/query_task",
            json={"img": obs_right},
        )
        resp.raise_for_status()
        raw = resp.json()
        return raw["task"], raw["round"]

    def reward_generation(self, obs_right):
        resp = requests.post(
            f"{self.gpt_server}/query_reward",
            json={"img": obs_right},
        )
        resp.raise_for_status()
        raw = resp.json()
        return raw["reward"]

    def get_offline_action_base(self, payload: Dict[Any, Any]):
        obs_left = payload["exterior_image_1_left"]
        obs_right = payload["exterior_image_2_left"]
        obs_wrist = payload["wrist_image_left"]
        eef_pose = payload["eef_position"]  # [x, y, z, qx, qy, qz, qw]
        gripper_position = payload["gripper_position"]
        left_resized, right_resized, wrist_resized = process_policy_images(obs_left, obs_right, obs_wrist)
        if payload["query_action_base"]:
            if eef_pose.ndim == 2 :
                eef_pose = eef_pose.squeeze(0)

            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(left_resized, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(right_resized, 224, 224),
                "observation/exterior_image_2_left": image_tools.resize_with_pad(wrist_resized, 224, 224),
                "observation/eef_position": eef_pose,
                "observation/gripper_position": gripper_position,
                "prompt": self.text,  # instruction
            }

            pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
            assert pred_action_chunk.shape == (50, 8) # 10,8
            action = pred_action_chunk[:20]
            action = action[::2]

            action = np.asarray(action).copy()
            action[:, -1] = (action[:, -1] > 0.8).astype(action.dtype)

            # Convert position deltas to absolute targets.
            action[:,:3] = action[:,:3] + eef_pose[:3]
            action = np.asarray(action, dtype=np.float32) 
        else:
            action = np.zeros((50,8))

        return json_response(action)

    def get_online_action_base(self, base_images: dict, eef_pose_rpy, gripper_position):
        left_resized = base_images["left"]
        right_resized = base_images["right"]
        wrist_resized = base_images["wrist"]

        eef_rpy = eef_pose_rpy[3:6]
        eef_quat = R.from_euler("xyz", eef_rpy, degrees=False).as_quat()
        eef_pose_quat = np.concatenate([eef_pose_rpy[:3], eef_quat], axis=-1)

        request_data = {
            "observation/exterior_image_1_left": left_resized,
            "observation/wrist_image_left": right_resized,
            "observation/exterior_image_2_left": wrist_resized,
            "observation/eef_position": eef_pose_quat,
            "observation/gripper_position": gripper_position,
            "prompt": self.text,
        }

        pred_action_chunk = self.pi05_client.infer(request_data)["actions"]
        assert pred_action_chunk.shape == (50, 8)

        action = pred_action_chunk[:20]
        action = action[::2]

        action = np.asarray(action).copy()
        action[:, -1] = (action[:, -1] > 0.8).astype(action.dtype)
        action[:, :3] = action[:, :3] + eef_pose_quat[:3]
        action = np.asarray(action, dtype=np.float32)
        for i in range(action.shape[0]):
            action[i] = self.action_scaler.scale(action[i])

        return action

    def get_current_base_action_step(self, action_base: np.ndarray) -> np.ndarray:
        action_base = np.asarray(action_base, dtype=np.float32)
        if action_base.ndim == 1:
            return action_base
        if action_base.ndim != 2:
            raise ValueError(f"action_base expected ndim 1 or 2, got shape={action_base.shape}")

        if self.args.use_base_chunk_first_action:
            return action_base[0]

        return action_base[0]

    # Obs / transition packaging
    def build_obs_packet(
        self,
        obs_left,
        obs_right,
        obs_wrist,
        eef_pose_rpy,
        gripper_position,
        action_base,
    ) -> dict:
        processed = process_policy_images(
            obs_left,
            obs_right,
            obs_wrist,
            base_size=224,
            residual_size=84,
        )

        left_base = processed["base"]["left"]
        right_base = processed["base"]["right"]
        wrist_base = processed["base"]["wrist"]

        left_residual = processed["residual"]["left"]
        right_residual = processed["residual"]["right"]
        wrist_residual = processed["residual"]["wrist"]

        eef_position_quat = rpy_pose_to_quat_pose(eef_pose_rpy)
        state = np.concatenate(
                    [
                        np.asarray(eef_position_quat, dtype=np.float32),
                        np.asarray(gripper_position, dtype=np.float32),
                    ],
                    axis=-1,
                ).astype(np.float32)
        state = self.state_standardizer.standardize(state)
        return {
            # 224x224 branch, useful for debug / consistency with base policy-side views
            "obs_left": np.asarray(left_base, dtype=np.uint8),
            "obs_right": np.asarray(right_base, dtype=np.uint8),
            "obs_wrist": np.asarray(wrist_base, dtype=np.uint8),

            "eef_position": np.asarray(eef_position_quat, dtype=np.float32),
            "gripper_position": np.asarray(gripper_position, dtype=np.float32),
            "action_base": None if action_base is None else np.asarray(action_base, dtype=np.float32),

            # 84x84 branch, matches residual actor / learner training input
            "train_obs": {
                "observation.images.exterior_image_1_left": hwc_to_chw_uint8(left_residual),
                "observation.images.exterior_image_2_left": hwc_to_chw_uint8(right_residual),
                "observation.images.wrist_image_left": hwc_to_chw_uint8(wrist_residual),
                "observation.state": state,
                "observation.base_action": None if action_base is None else np.asarray(action_base, dtype=np.float32),
            },

            # keep processed base images for pi05 if you want to reuse without recomputing
            "base_images": {
                "left": np.asarray(left_base, dtype=np.uint8),
                "right": np.asarray(right_base, dtype=np.uint8),
                "wrist": np.asarray(wrist_base, dtype=np.uint8),
            },
        }

    def build_transition_payload(
        self,
        curr_obs_packet: dict,
        combined_action_quat: np.ndarray,
        next_obs_packet: dict,
        reward: float,
        done: bool,
    ) -> dict:
        return {
            "obs": {
                "observation.images.exterior_image_1_left": curr_obs_packet["train_obs"]["observation.images.exterior_image_1_left"].tolist(),
                "observation.images.exterior_image_2_left": curr_obs_packet["train_obs"]["observation.images.exterior_image_2_left"].tolist(),
                "observation.images.wrist_image_left": curr_obs_packet["train_obs"]["observation.images.wrist_image_left"].tolist(),
                "observation.state": curr_obs_packet["train_obs"]["observation.state"].tolist(),
                "observation.base_action": (
                    None
                    if curr_obs_packet["train_obs"]["observation.base_action"] is None
                    else curr_obs_packet["train_obs"]["observation.base_action"].tolist()
                ),
            },
            "action": np.asarray(combined_action_quat, dtype=np.float32).tolist(),
            "next_obs": {
                "observation.images.exterior_image_1_left": next_obs_packet["train_obs"]["observation.images.exterior_image_1_left"].tolist(),
                "observation.images.exterior_image_2_left": next_obs_packet["train_obs"]["observation.images.exterior_image_2_left"].tolist(),
                "observation.images.wrist_image_left": next_obs_packet["train_obs"]["observation.images.wrist_image_left"].tolist(),
                "observation.state": next_obs_packet["train_obs"]["observation.state"].tolist(),
                "observation.base_action": (
                    None
                    if next_obs_packet["train_obs"]["observation.base_action"] is None
                    else next_obs_packet["train_obs"]["observation.base_action"].tolist()
                ),
            },
            "reward": float(reward),
            "done": bool(done),
        }

    def build_tranistion_payload(self, trajectory) -> dict:
        return {
            "trajectory": trajectory,
            "meta": {
                "round": int(self.round),
                "task": self.text,
                "length": len(trajectory),
                "timestamp": time.time(),
                "policy_version": int(self.policy_version),
            },
        }

    # Pushing / syncing
    def push_transition(self,trajectory: List[dict], warm_up = False):
        if not self.args.push_trajectories_to_learner:
            return
        if len(trajectory) == 0:
            return

        payload = self.build_tranistion_payload(trajectory)
        if warm_up:
            resp = self.learner_client.push_warmup_trajectory(payload)
        else:
            resp = self.learner_client.push_transition(payload)
        print(f"[collector] pushed trajectory #{self.total_trajectory_count} to learner, resp={resp}")

        return resp.json()

    def maybe_pull_latest_policy(self):
        if self.total_transition_count % max(1, self.args.policy_poll_interval_steps) != 0:
            return

        try:
            resp = self.learner_client.get_latest_policy(self.policy_version)
        except Exception as e:
            print(f"[collector] failed to query learner policy: {e}")
            return

        if not resp.get("has_update", False):
            return

        try:
            policy_state_b64 = resp["policy_state_b64"]
            payload = decode_policy_payload_from_base64(policy_state_b64)
            self.actor_policy.load_policy_payload(payload, strict=self.args.actor_sync_strict)
            self.policy_version = int(payload["version"])
            self.actor_policy.version = self.policy_version
            print(f"[collector] updated residual actor to version {self.policy_version}")
        except Exception as e:
            print(f"[collector] failed to load learner policy: {e}")

    # Environment interaction
    def reset(self, compute_prev_reward: bool = True) -> dict:
        self.t_step = 0
        self.env.reset()

        obs = _extract_observation(
            self.args,
            self.env.get_observation(),
            save_to_disk=False,
        )
        obs_left = obs["left_image"]
        obs_right = obs["right_image"]
        obs_wrist = obs["wrist_image"]
        eef_pose_rpy = obs["cartesian_position"]
        gripper_position = obs["gripper_position"]

        if self.round != 0 and compute_prev_reward:
            self.reward = float(self.reward_generation(obs_right))
            print("reward:", self.reward)
            time.sleep(5)

        print(f"---------------------------- trajectory {self.round} ----------------------------")
        self.text, self.round = self.task_generation(obs_right)
        print("task:", self.text)

        processed = process_policy_images(
            obs_left,
            obs_right,
            obs_wrist,
            base_size=224,
            residual_size=84,
        )

        action_base = self.get_online_action_base(
            processed["base"],
            eef_pose_rpy,
            gripper_position,
        )

        obs_packet = self.build_obs_packet(
            obs_left=obs_left,
            obs_right=obs_right,
            obs_wrist=obs_wrist,
            eef_pose_rpy=eef_pose_rpy,
            gripper_position=gripper_position,
            action_base=action_base,
        )
        self.current_obs_packet = obs_packet
        self.current_trajectory = []

        return obs_packet

    def combine_action(self, obs_packet: dict, residual_action: np.ndarray) -> np.ndarray:
        base_action = obs_packet["action_base"]
        if base_action is None:
            raise ValueError("obs_packet.action_base is None")

        base_action_step = self.get_current_base_action_step(base_action)
        residual_action = np.asarray(residual_action, dtype=np.float32).reshape(-1)

        if residual_action.shape[0] != self.args.action_dim:
            raise ValueError(
                f"residual_action shape mismatch: expected ({self.args.action_dim},), "
                f"got {residual_action.shape}"
            )

        combined_action = base_action_step + residual_action
        combined_action = self.action_scaler.unscale(combined_action)
        combined_action = np.asarray(combined_action, dtype=np.float32)
        return combined_action

    def step_once(self, combined_action_quat: np.ndarray):
        env_action = quat_action_to_rpy_action(combined_action_quat, self.args.z_min)

        self.env.step(env_action)
        self.t_step += 1

        done = bool(self.t_step >= self.args.max_timesteps - 1)

        next_raw_obs = _extract_observation(
            self.args,
            self.env.get_observation(),
            save_to_disk=False,
        )
        next_obs_left = next_raw_obs["left_image"]
        next_obs_right = next_raw_obs["right_image"]
        next_obs_wrist = next_raw_obs["wrist_image"]
        next_eef_pose_rpy = next_raw_obs["cartesian_position"]
        next_gripper_position = next_raw_obs["gripper_position"]

        if done:
            reward = float(self.reward_generation(next_obs_right))
            self.reward = reward
            next_action_base = None
        else:
            reward = 0.0
            processed = process_policy_images(
                next_obs_left,
                next_obs_right,
                next_obs_wrist,
                base_size=224,
                residual_size=84,
            )

            next_action_base = self.get_online_action_base(
                processed["base"],
                next_eef_pose_rpy,
                next_gripper_position,
            )

        next_obs_packet = self.build_obs_packet(
            obs_left=next_obs_left,
            obs_right=next_obs_right,
            obs_wrist=next_obs_wrist,
            eef_pose_rpy=next_eef_pose_rpy,
            gripper_position=next_gripper_position,
            action_base=next_action_base,
        )

        return next_obs_packet, reward, done

    # Main collector loop
    def collector_loop(self):
        obs_packet = self.reset()

        while not self.online_move:
            if not self.online_warmup:
                continue
            step_start = time.time()

            actor_obs = {
                "observation.images.exterior_image_1_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.exterior_image_1_left"], device=self.device
                ),
                "observation.images.exterior_image_2_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.exterior_image_2_left"], device=self.device
                ),
                "observation.images.wrist_image_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.wrist_image_left"], device=self.device
                ),
                "observation.state": torch.as_tensor(
                    obs_packet["train_obs"]["observation.state"], dtype=torch.float32, device=self.device
                ),
                "observation.base_action": torch.as_tensor(
                    obs_packet["train_obs"]["observation.base_action"], dtype=torch.float32, device=self.device
                ),
            }

            rand_actions = (  # line2: Sample noise εt ∼ U (−noise scale, noise scale)
                    torch.rand((args.num_envs, args.action_dim), device=args.device) * 2 - 1
                ) * 0.02
            residual_action = rand_actions.detach().cpu().numpy().astype(np.float32)

            combined_action = self.combine_action(obs_packet, residual_action)

            next_obs_packet, reward, done = self.step_once(combined_action)

            transition_payload = self.build_transition_payload(
                curr_obs_packet=obs_packet,
                combined_action_quat=combined_action,
                next_obs_packet=next_obs_packet,
                reward=reward,
                done=done,
            )


            if done:
                self.total_trajectory_count += 1
                obs_packet = self.reset(compute_prev_reward=False)
            else:
                obs_packet = next_obs_packet
            warm_up = True
            resp = self.push_transition(transition_payload, warm_up)
            ok = resp["ok"]
            if ok == False:
                break

            if self.args.control_hz > 0:
                dt = 1.0 / self.args.control_hz
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            print(
                f"[collector] warm up step={self.t_step} "
                f"traj_len={len(self.current_trajectory)} "
                f"reward={reward:.3f} done={done} "
                f"policy_version={self.policy_version}"
            )

        self.env.reset()

        while True:
            if not self.online_move:
                continue
            step_start = time.time()
            self.maybe_pull_latest_policy()

            actor_obs = {
                "observation.images.exterior_image_1_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.exterior_image_1_left"], device=self.device
                ),
                "observation.images.exterior_image_2_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.exterior_image_2_left"], device=self.device
                ),
                "observation.images.wrist_image_left": torch.as_tensor(
                    obs_packet["train_obs"]["observation.images.wrist_image_left"], device=self.device
                ),
                "observation.state": torch.as_tensor(
                    obs_packet["train_obs"]["observation.state"], dtype=torch.float32, device=self.device
                ),
                "observation.base_action": torch.as_tensor(
                    obs_packet["train_obs"]["observation.base_action"], dtype=torch.float32, device=self.device
                ),
            }

            residual_action = self.actor_policy.act(actor_obs)
            residual_action = residual_action.detach().cpu().numpy().astype(np.float32)

            combined_action = self.combine_action(obs_packet, residual_action)

            next_obs_packet, reward, done = self.step_once(combined_action)

            transition_payload = self.build_transition_payload(
                curr_obs_packet=obs_packet,
                combined_action_quat=combined_action,
                next_obs_packet=next_obs_packet,
                reward=reward,
                done=done,
            )


            if done:
                self.total_trajectory_count += 1
                obs_packet = self.reset(compute_prev_reward=False)
            else:
                obs_packet = next_obs_packet
            self.push_transition(transition_payload)

            if self.args.control_hz > 0:
                dt = 1.0 / self.args.control_hz
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            print(
                f"[collector] step={self.t_step} "
                f"traj_len={len(self.current_trajectory)} "
                f"reward={reward:.3f} done={done} "
                f"policy_version={self.policy_version}"
            )


# Main
def main(args: Args):
    cfg = ResidualTD3FrankaTomatoConfig()
    cfg.agent.device = str(
        torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    )

    with prevent_keyboard_interrupt():
        env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
        robot_state, _ = env.get_state()

        print("Initial joint positions:", robot_state.get("joint_positions", "N/A"))

        gpt_server = f"http://{args.gpt_host}:{args.gpt_port}"
        pi05_client = websocket_client_policy.WebsocketClientPolicy(args.pi05_host, args.pi05_port)
        learner_client = LearnerClient(args.learner_host, args.learner_port, args.learner_request_timeout_s)

        image_keys = list(cfg.rl_camera)

        actor_policy = ResidualActorPolicy(
            obs_shape=(3, 84, 84),
            prop_shape=(8,),
            action_dim=args.action_dim,
            rl_cameras=image_keys,
            cfg=cfg.agent,   # Must match the training agent configuration.
        )

        collector = Collector(
            env=env,
            args=args,
            gpt_server=gpt_server,
            pi05_client=pi05_client,
            learner_client=learner_client,
            actor_policy=actor_policy,
        )

        collector.collector_loop()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

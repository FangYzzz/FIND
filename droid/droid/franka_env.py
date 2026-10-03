from copy import deepcopy

import gym
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import os
import numpy as np
import time

from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame
import requests


class FrankaRobotClient:
    def __init__(self, ip="127.0.0.1", port=8006):
        self.base_url = f"http://{ip}:{port}"


    def update_joints(self, joints, velocity=False, blocking=False, cartesian_noise=None):
        r = requests.post(
            f"{self.base_url}/update_joints",
            json={
                "command": list(joints),
                "velocity": velocity,
                "blocking": blocking,
                "cartesian_noise": cartesian_noise if cartesian_noise is not None else None,
            },
        )
        r.raise_for_status()
        return r.json()


    def update_gripper(self, gripper_action, velocity=False, blocking=False):
        r = requests.post(
            f"{self.base_url}/update_gripper",
            json={
                "command": gripper_action,
                "velocity": velocity,
                "blocking": blocking,
            },
        )
        r.raise_for_status()
        return r.json()

    def update_command(
        self,
        action,
        action_space="cartesian_velocity",
        gripper_action_space=None,
        blocking=False,
    ):
        payload = {
            "command": np.asarray(action).tolist(),
            "action_space": action_space,
            "gripper_action_space": gripper_action_space,
            "blocking": blocking,
        }
        r = requests.post(f"{self.base_url}/update_command", json=payload)
        r.raise_for_status()
        return r.json()["result"]

    def get_robot_state(self):
        r = requests.get(f"{self.base_url}/get_robot_state")
        r.raise_for_status()
        data = r.json()
        return data["state_dict"], data["timestamp_dict"]

    def create_action_dict(self, action):
        payload = {
            "action": np.asarray(action).tolist(),
            "action_space": "cartesian_velocity",
            "gripper_action_space": None,
            "blocking": False,
        }
        r = requests.post(f"{self.base_url}/create_action_dict", json=payload)
        r.raise_for_status()
        return r.json()["result"]


class RobotEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True):
        # Initialize Gym Environment
        super().__init__()

        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.check_action_range = "velocity" in action_space

        # Robot Configuration
        # [-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9]
        self.reset_joints = np.array([-0.3, -1 / 5 * np.pi, 0, -3.7 / 5 * np.pi, 0, 2.7 / 5 * np.pi, 2.2])


        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7 if ("cartesian" in action_space) else 8
        self.control_hz = 15

        if nuc_ip is None:
            self._robot = FrankaRobotClient()


        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        # Return Action Info
        return action_info

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict


    def get_observation(self):
        obs_dict = {"timestamp": {}}

        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)

        intrinsics = {}

        return obs_dict

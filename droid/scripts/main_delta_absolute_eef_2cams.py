# ruff: noqa


import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
import cv2
from scipy.spatial.transform import Rotation as R
import copy
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional
import os
import numpy as np
from franky import *
import time

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "24285872"
    right_camera_id: str = None # "<your_camera_id>"
    wrist_camera_id: str = "11022812"  # "11022812"

    # Policy parameters
    external_camera: Optional[str] = (
        # None  # which external camera should be fed to the policy, choose from ["left", "right"]
        "left"
    )

    # Rollout parameters
    max_timesteps: int = 140
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 45

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8008  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
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


def prepare_image_256(img, size=(224, 224)):
    """Center-crop to a square and resize to an RGB uint8 image."""
    if img is None:
        return None
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)

    return out.astype(np.uint8, copy=False)


def main(args: Args):

    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    robot_state,_ = env.get_state()
    print(robot_state["cartesian_position"])
    eef_q = copy.deepcopy(robot_state["cartesian_position"][3:6])
    gripper = None
    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video_wirst = []
        video_left = []
        video_right = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                # and improve latency.
                right_image = prepare_image_256(curr_obs["right_image"])
                wrist_image = prepare_image_256(curr_obs["wrist_image"])
                video_right.append(right_image)
                video_wirst.append(wrist_image)

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    robot_state_,_ = env.get_state()
                    eef_state = copy.deepcopy(robot_state_["cartesian_position"])
                    eef_pose = curr_obs["cartesian_position"]
                    eef_rpy = eef_pose[3:6]
                    eef_quat = R.from_euler('xyz', eef_rpy, degrees=False).as_quat()
                    eef_pose = np.concatenate([eef_pose[:3], eef_quat], axis=-1)
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/wrist_image_left": image_tools.resize_with_pad(wrist_image, 224, 224),
                        "observation/exterior_image_2_left": image_tools.resize_with_pad(right_image, 224, 224),
                        "observation/eef_position": eef_pose,
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]

                    assert pred_action_chunk.shape == (50, 8) # 10,8

                # Select current action to execute from chunk

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                    gripper = np.ones((1,))
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                    gripper = np.zeros((1,))

                # clip all dimensions of action to [-1, 1]

                R_state = R.from_euler('xyz', eef_state[3:6]).as_matrix()
                q_action = action[3:7] 
                norm = np.linalg.norm(q_action, axis=-1, keepdims=True)
                q_action = q_action / np.clip(norm, 1e-12, None)
                sign = np.where(q_action[..., 3:4] < 0, -1.0, 1.0)
                q_action = q_action * sign
                R_delta = R.from_quat(q_action).as_matrix()
                rpy_cmd = R.from_matrix(R_delta).as_euler('xyz', degrees=False)

                action[:3] = action[:3] + eef_state[:3]
                # Enforce the end-effector floor limit.
                print("action[2]: ", action[2])
                action[3:6] = rpy_cmd
                action_ = np.concatenate([action[:3], rpy_cmd, gripper],axis=-1)

                # Prevent touching the table
                env.step(action_)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break




        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if "right_cam" in key:
            right_image = image_observations[key]
        elif "wrist_cam" in key:
            wrist_image = image_observations[key]

    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")


    return {
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "gripper_position": gripper_position,
    }

def save_np_image(img: np.ndarray, path: str):
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3

    os.makedirs(os.path.dirname(path), exist_ok=True)

    Image.fromarray(img).save(path)

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

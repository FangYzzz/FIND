import os
import random
from collections import defaultdict

from droid.camera_utils.camera_readers.zedx_camera import gather_zed_cameras  # ZEDX
from droid.camera_utils.info import get_camera_type


class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}):
        stream_configs = [  # ZEDXmini
            {
                "name": "left_cam",
                "ip": "192.168.55.1",
                "port": 30004,
                "is_hand_camera": False,
            },
            {
                "name": "wrist_cam",
                "ip": "192.168.55.1",
                "port": 30000,
                "is_hand_camera": True,
            },
            {
                "name": "right_cam",
                "ip": "192.168.55.1",
                "port": 30002,
                "is_hand_camera": False,
            },
        ]
        zed_cameras = gather_zed_cameras(stream_configs=stream_configs)

        self.camera_dict = {cam.serial_number: cam for cam in zed_cameras}
        for cam_id, cam in self.camera_dict.items():  # ZEDX
            curr_cam_kwargs = camera_kwargs.get(cam_id, {})
            cam.set_reading_parameters(**curr_cam_kwargs)

        self.set_trajectory_mode()

    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        close_all = any([cam.high_res_calibration for cam in self.camera_dict.values()])

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):


        pass

    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    def read_cameras(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue
            data_dict = self.camera_dict[cam_id].read_camera()

            for key in data_dict:

                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict, full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.disable_camera()

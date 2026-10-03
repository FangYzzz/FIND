from copy import deepcopy

import cv2
import numpy as np

from droid.misc.parameters import hand_camera_id
from droid.misc.time import time_ms

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    sl = None
    print("WARNING: You have not setup the ZEDX cameras, and currently cannot use them")


resize_func_map = {"cv2": cv2.resize, None: None}


if sl is not None:
    standard_params = dict(
        depth_minimum_distance=0.1,
        camera_resolution=sl.RESOLUTION.HD720,
        depth_stabilization=False,
        camera_fps=60,
        camera_image_flip=sl.FLIP_MODE.OFF,
    )

    advanced_params = dict(
        depth_minimum_distance=0.1,
        camera_resolution=sl.RESOLUTION.HD2K,
        depth_stabilization=False,
        camera_fps=15,
        camera_image_flip=sl.FLIP_MODE.OFF,
    )
else:
    standard_params = {}
    advanced_params = {}


def gather_zed_cameras(stream_configs):
    if sl is None:
        raise RuntimeError("pyzed.sl is not available")

    all_zed_cameras = []
    print("stream_configs: ", stream_configs)
    for cfg in stream_configs:
        cam = StreamZedCamera(
            name=cfg["name"],
            stream_ip=cfg["ip"],
            stream_port=cfg["port"],
            is_hand_camera=cfg.get("is_hand_camera", False),
        )
        all_zed_cameras.append(cam)

    return all_zed_cameras


class StreamZedCamera:
    def __init__(self, name, stream_ip, stream_port, is_hand_camera=False):
        self.name = name
        self.sl = sl  # Store module for later use
        self.zed = self.sl.Camera()
        self.init_params = self.sl.InitParameters()
        self.runtime_params = self.sl.RuntimeParameters()
        self.image = self.sl.Mat()
        self.depth = self.sl.Mat()
        self.close = 0.1
        self.far = 3.0
        self.started = False   
        self.serial_number = None

        self._intrinsics={}

        self._current_params = None
        self._extrinsics = {}

        self.close_depth = 0.1
        self.far_depth = 3.0

        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.pointcloud = False
        self.resize_func = None

        self.current_mode = None
        self.stop()
        self.start(stream_ip, stream_port)
        self.skip_reading = False
        print(f"Opening Stream ZED: {self.name} @ {stream_ip}:{stream_port}")

    def enable_advanced_calibration(self):
        self.high_res_calibration = True
    def stop(self):
        self.zed.close()
        self.started = False
    def start(self, stream_ip = '192.168.55.1', stream_port = 30000, mode="standard"):
        # initial params
        self.init_params.coordinate_units = self.sl.UNIT.METER
        self.init_params.set_from_stream(stream_ip, stream_port)
        self.init_params.depth_mode = self.sl.DEPTH_MODE.NEURAL
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 60
        if mode == "standard":
            params = standard_params
        elif mode == "advanced":
            params = advanced_params

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        fx, fy = calib.left_cam.fx, calib.left_cam.fy
        cx, cy = calib.left_cam.cx, calib.left_cam.cy
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Grab a frame to get original size
        if self.zed.grab(self.runtime_params) == self.sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)
            rgb = self.image.get_data()[:, :, :3]
            W, H = rgb.shape[1], rgb.shape[0]
        cam_info = self.zed.get_camera_information()
        self.serial_number = cam_info.serial_number
        print(f"""Camera Parameters:
            - Resolution : {rgb.shape}
            - Depth mode : {self.init_params.depth_mode}
            - Intrinsic  : {self.intrinsic}
            """)
        self.started = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    def set_calibration_mode(self):
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.zed_resolution = sl.Resolution(0, 0)
        self.resizer_resolution = (0, 0)

        if self.high_res_calibration:
            init_params = dict(
                depth_minimum_distance=0.1,
                camera_resolution=sl.RESOLUTION.HD2K,
                depth_stabilization=False,
                camera_fps=15,
                camera_image_flip=sl.FLIP_MODE.OFF,
            )
        else:
            init_params = dict(
                depth_minimum_distance=0.1,
                camera_resolution=sl.RESOLUTION.HD720,
                depth_stabilization=False,
                camera_fps=60,
                camera_image_flip=sl.FLIP_MODE.OFF,
            )

        if self._current_params != init_params:
            self._configure_camera(init_params)

        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.zed_resolution = sl.Resolution(*self.traj_resolution)
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = self.traj_resolution

        init_params = dict(
            depth_minimum_distance=0.1,
            camera_resolution=sl.RESOLUTION.HD720,
            depth_stabilization=False,
            camera_fps=60,
            camera_image_flip=sl.FLIP_MODE.OFF,
        )

        if self._current_params != init_params:
            self._configure_camera(init_params)

        self.current_mode = "trajectory"

    def _configure_camera(self, init_params):
        self.disable_camera()

        self._cam = sl.Camera()
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._left_depth = sl.Mat()
        self._right_depth = sl.Mat()
        self._left_pointcloud = sl.Mat()
        self._right_pointcloud = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self._runtime.confidence_threshold = 50 
        self._runtime.texture_confidence_threshold = 100
        self._current_params = init_params
        sl_params = sl.InitParameters(**init_params)
        sl_params.coordinate_units = sl.UNIT.METER
        sl_params.set_from_stream(self.stream_ip, self.stream_port)
        sl_params.depth_mode = sl.DEPTH_MODE.NEURAL
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF

        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Stream camera failed to open: {self.name} @ {self.stream_ip}:{self.stream_port}, err={status}"
            )

        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(calib_params.left_cam),
            self.serial_number + "_right": self._process_intrinsics(calib_params.right_cam),
        }

        self.latency = int(2.5 * (1e3 / sl_params.camera_fps))

    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array(
            [[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]],
            dtype=np.float32,
        )
        intrinsics["distCoeffs"] = np.array(list(params.disto), dtype=np.float32)
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        assert filename.endswith(".svo")
        recording_param = sl.RecordingParameters(filename, sl.SVO_COMPRESSION_MODE.H265)
        err = self._cam.enable_recording(recording_param)
        assert err == sl.ERROR_CODE.SUCCESS

    def stop_recording(self):
        self._cam.disable_recording()

    def _process_frame(self, frame):
        frame = np.array(frame.get_data(), copy=True)
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self):

        err = self.zed.grab(self.runtime_params)


        data_dict = {}

        self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)

        rgb = self.image.get_data()[:, :, :3]
        rgb = rgb[..., ::-1]  # Convert BGR to RGB

        # Ensure standard ndarray for OpenCV 4.13+ (ZED get_data may return array OpenCV rejects)
        rgb = np.array(rgb, dtype=np.uint8, copy=True, order='C')
        # resize

        # crop close and far

        data_dict["image"] = {self.name: rgb}
        return data_dict #, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "_cam"):
            self._current_params = None
            self._cam.close()
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

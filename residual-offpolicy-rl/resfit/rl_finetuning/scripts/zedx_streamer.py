import pyzed.sl as sl
import numpy as np

class ZEDStreamer:
    def __init__(self):
        self.sl = sl  # Store module for later use
        self.zed = self.sl.Camera()
        self.init_params = self.sl.InitParameters()
        self.runtime_params = self.sl.RuntimeParameters()
        self.image = self.sl.Mat()
        self.depth = self.sl.Mat()
        self.close = 0.1
        self.far = 3.0
        self.started = False


    def start(self, stream_ip = '192.168.55.1', stream_port = 30000):
        # initial params
        self.init_params.coordinate_units = self.sl.UNIT.METER
        self.init_params.set_from_stream(stream_ip, stream_port)
        self.init_params.depth_mode = self.sl.DEPTH_MODE.NEURAL

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

        print(f"""Camera Parameters:
          - Resolution : {rgb.shape}
          - Depth mode : {self.init_params.depth_mode}
          - Intrinsic  : {self.intrinsic}
        """)
        self.started = True

    def get_frame(self):
        self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)

        rgb = self.image.get_data()[:, :, :3]
        rgb = rgb[..., ::-1]  # Convert BGR to RGB
        depth = self.depth.get_data()

        # Ensure standard ndarray for OpenCV 4.13+ (ZED get_data may return array OpenCV rejects)
        rgb = np.array(rgb, dtype=np.uint8, copy=True, order='C')
        depth = np.array(depth, dtype=np.float32, copy=True, order='C')
        # resize

        # crop close and far
        depth[(depth < self.close) | (depth > self.far) | np.isnan(depth)] = 0.

        return rgb, depth

    def get_status(self):
        return self.started and (self.zed.grab(self.runtime_params) == self.sl.ERROR_CODE.SUCCESS)

    def stop(self):
        self.zed.close()
        self.started = False



import os
from cv2 import aruco

nuc_ip =None #"0.0.0.0" # ""
robot_ip = "172.17.0.2"
laptop_ip = "127.0.1.1"
sudo_password = "F990123y"
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = ""

hand_camera_id = "24285872"
varied_camera_1_id = "11022812"
varied_camera_2_id = None  # "29931811"

CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

ubuntu_pro_token = ""

droid_version = "1.3"

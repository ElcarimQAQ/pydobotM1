from real_world.camera.kinect import KinectClient
from real_world.camera.realsense import RealSense
from real_world.realur5 import UR5
from real_world.gripper.wsg50 import WSG50
from real_world.gripper.rg2 import RG2
from real_world.m1 import M1
from serial.tools import list_ports
import numpy as np


DEFAULT_ORN = [32]
DIST_UR5 = 1.34
WORKSPACE_SURFACE = 0.125
MIN_GRASP_WIDTH = 1
MAX_GRASP_WIDTH = 50
MIN_UR5_BASE_SAFETY_RADIUS = 0.3
# workspace pixel crop
WS_PC = [100, -165, 420, -405]


UR5_VELOCITY = 0.5
UR5_ACCELERATION = 0.3

CLOTHS_DATASET = {
    'hannes_tshirt': {
        'flatten_area': 0.0524761,
        'cloth_size': (0.45, 0.55),
        'mass': 0.2
    },
}
CURRENT_CLOTH = 'hannes_tshirt'

MIN_M1_BASE_SAFETY_RADIUS = 0.2
MAX_M1_BASE_SAFETY_RADIUS = 0.4

WORKSPACE_BOUNDS = np.array([
    [0.150, 0.400],
    [-0.315, 0.315],
    [0.050, 0.220]
])  # x, y, z

def get_ur5s():
    return [
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=RG2(tcp_ip='XXX.XXX.X.XXX')),
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=WSG50(tcp_ip='XXX.XXX.X.XXX')),
    ]

def get_m1(run=True):
    if not run:
        return None
    ports = list_ports.comports()
    return M1(port=ports[2].device, verbose=True)


def get_top_cam():
    # server:
    # return KinectClient('49.235.98.247', 7778)
    # local:
    return KinectClient('127.0.0.1', 1111)


def get_front_cam():
    return RealSense(
        tcp_ip='127.0.0.1',
        tcp_port=12345,
        im_h=720,
        im_w=1280,
        max_depth=3.0)

# test
if __name__ == '__main__':
    print(list_ports.comports()[2].device)
    m1 = get_m1()
    m1.out_of_the_way()
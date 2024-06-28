import numpy as np
from pydobot.enums.CommunicationProtocolIDs import CommunicationProtocolIDs
from pydobot.enums.ControlValues import ControlValues
from pydobot.message import Message
from pydobot.enums import PTPMode
from time import sleep
from pydobot import Dobot



class M1(Dobot):
    # HOME = [0, 0, np.pi / 2, 0]
    HOME = [0.221, 0.315, 0.230, 0]

    RESET = [0.370, -0.40, 0.200, 0]


    def __init__(self, port, verbose=False,
                 velocity=30,
                 acceleration=30,
                 gripper=0
                 ):
        super(M1, self).__init__(port, verbose)
        self._Clear_All_Alarm_States()
        # speeds of velocity along x, y, z, r axes acceleration along x, y, z, r axes Max=100, Min=0
        self._set_ptp_joint_params(50, 50, 50, 50, 50, 50, 50, 100)
        self._set_ptp_coordinate_params(velocity, acceleration)
        self._set_arm_orientation('L')
        self.grip(gripper)
        self.ee_tip_z_offset = 0.12


    def movej(self, **kwargs):
        return self.move('j', **kwargs)

    def movel(self, **kwargs):
        return self.move('l', **kwargs)

    def _set_home_cmd(self):
        msg = Message()
        msg.id = CommunicationProtocolIDs.SET_HOME_CMD
        msg.ctrl = ControlValues.THREE
        msg.params = bytearray(bytes([0x01]))
        msg.params.append(0x00)
        return self._send_command(msg)

    def homej(self, **kwargs):
        self.movej(params=self.RESET, **kwargs)

    def check_pose_reachable(self, pose):
        from real_world.setup import MIN_M1_BASE_SAFETY_RADIUS, MAX_M1_BASE_SAFETY_RADIUS, WORKSPACE_BOUNDS
        if pose[0] < WORKSPACE_BOUNDS[0, 0] or pose[0] > WORKSPACE_BOUNDS[0, 1]:
            return False
        if pose[1] < WORKSPACE_BOUNDS[1, 0] or pose[1] > WORKSPACE_BOUNDS[1, 1]:
            return False
        if pose[2] < WORKSPACE_BOUNDS[2, 0] or pose[2] > WORKSPACE_BOUNDS[2, 1]:
            return False
        norm = np.linalg.norm(pose[:2])
        return MAX_M1_BASE_SAFETY_RADIUS > norm > MIN_M1_BASE_SAFETY_RADIUS

    def move(self, move_type, params,
             blocking=True,
             j_acc=None, j_vel=None,
             times=0.0, blend=0.0,
             clear_state_history=False, use_pos=False):

        # If blocking call, pause until robot stops moving
        for i in range(3):
            params[i] *= 1000

        if move_type == 'l':
            self._set_ptp_cmd(params[0], params[1], params[2], params[3], mode=PTPMode.MOVL_XYZ, wait=blocking)
        elif move_type == 'j':
            self._set_ptp_cmd(params[0], params[1], params[2], params[3], mode=PTPMode.MOVJ_XYZ, wait=blocking)
        return True

    def close_grippers(self, blocking=True):
        self.suck(1)
        # self.grip(0)
        if blocking:
            sleep(1)

    def open_grippers(self, blocking=True):
        self.grip(1)
        self.suck(0)
        if blocking:
            sleep(1)

    def out_of_the_way(self):
        self.movel(
            params=[0.221, 0.315, 0.220, 32],
            blocking=True,
            use_pos=True)

    def reset(self):
        self.homej()


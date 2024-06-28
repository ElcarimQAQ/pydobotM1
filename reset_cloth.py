import matplotlib.pyplot as plt

from real_world.utils import (
    get_cloth_mask,
    pix_to_3d_position,
    pick_place_primitive_helper,
    InvalidDepthException,
    bound_grasp_pos
)
from real_world.setup import (
    DEFAULT_ORN
)
import numpy as np
import random
from serial.tools import list_ports
from real_world.setup import get_m1
from real_world.camera.kinect import KinectClient
import cv2

def pick_and_drop_m1(m1, top_camera, top_cam_m1_pose, cam_depth_scale):
    before_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
    rgb, depth = top_camera.get_rgbd()
    cloth_mask = get_cloth_mask(rgb=rgb)
    plt.imshow(cloth_mask)
    plt.show()

    points = np.array(np.where(cloth_mask == 1))

    # Find random point on cloth
    indices = list(range(points.shape[1]))
    random.shuffle(indices)
    for i in indices:
        point = points[:, i]
        y, x = point

        # Try with arm
        try:
            pick_pos = list(pix_to_3d_position(
                x=x, y=y, depth_image=depth,
                cam_intr=top_camera.color_intr,
                cam_extr=top_cam_m1_pose,
                cam_depth_scale=cam_depth_scale))
            pick_pos = bound_grasp_pos(pick_pos)

            if m1.check_pose_reachable(pose=list(pick_pos) + list(DEFAULT_ORN)):
                cv2.circle(rgb, (x, y), 5, (0, 0, 255), -1)
                plt.imshow(rgb)
                plt.show()

                pick_place_primitive_helper(arm=m1, pick_pose=list(pick_pos) + list(DEFAULT_ORN),
                place_pose=[0.3, 0.1, 0.125] + DEFAULT_ORN)
                m1.out_of_the_way()
                after_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
                intersection = np.logical_and(before_mask, after_mask).sum()
                union = np.logical_or(before_mask, after_mask).sum()
                iou = intersection/union
                if iou < 1 - 2e-1:
                    return
                else:
                    continue
        except InvalidDepthException as e:
            pass

    m1.out_of_the_way()

def pick_and_drop(ur5_pair, top_camera, top_cam_right_ur5_pose,
                  top_cam_left_ur5_pose, cam_depth_scale):
    before_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
    rgb, depth = top_camera.get_rgbd()
    cloth_mask = get_cloth_mask(rgb=rgb)
    points = np.array(np.where(cloth_mask == 1))

    # Find random point on cloth
    indices = list(range(points.shape[1]))
    random.shuffle(indices)
    for i in indices:
        point = points[:, i]
        y, x = point

        # Try with right arm
        try:
            pick_pos = list(pix_to_3d_position(
                x=x, y=y, depth_image=depth,
                cam_intr=top_camera.color_intr,
                cam_extr=top_cam_right_ur5_pose,
                cam_depth_scale=cam_depth_scale))
            pick_pos = bound_grasp_pos(pick_pos)

            if ur5_pair.right_ur5.check_pose_reachable(
                pose=list(pick_pos) + list(DEFAULT_ORN))\
                and pick_place_primitive_helper(
                    ur5=ur5_pair.right_ur5,
                    pick_pose=list(pick_pos) + list(DEFAULT_ORN),
                    place_pose=[0.65, 0.1, 0.35] + DEFAULT_ORN):
                ur5_pair.out_of_the_way()
                after_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
                intersection = np.logical_and(before_mask, after_mask).sum()
                union = np.logical_or(before_mask, after_mask).sum()
                iou = intersection/union
                if iou < 1 - 2e-1:
                    return
                else:
                    continue
        except InvalidDepthException as e:
            pass

        # Try with left arm
        try:
            pick_pos = list(pix_to_3d_position(
                x=x, y=y, depth_image=depth,
                cam_intr=top_camera.color_intr,
                cam_extr=top_cam_left_ur5_pose,
                cam_depth_scale=cam_depth_scale))
            pick_pos = bound_grasp_pos(pick_pos)

            if ur5_pair.left_ur5.check_pose_reachable(
                pose=list(pick_pos) + list(DEFAULT_ORN))\
                and pick_place_primitive_helper(
                    ur5=ur5_pair.left_ur5,
                    pick_pose=list(pick_pos) + list(DEFAULT_ORN),
                    place_pose=[0.65, 0.1, 0.35] + DEFAULT_ORN):
                ur5_pair.out_of_the_way()
                after_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
                intersection = np.logical_and(before_mask, after_mask).sum()
                union = np.logical_or(before_mask, after_mask).sum()
                iou = intersection/union
                if iou < 1 - 2e-1:
                    return
                else:
                    continue
        except InvalidDepthException:
            pass

    ur5_pair.out_of_the_way()

if __name__ == '__main__':
    print(list_ports.comports()[2].device)
    m1 = get_m1()
    m1._Clear_All_Alarm_States()
    m1.out_of_the_way()
    cam = KinectClient('127.0.0.1', 1111)
    top_cam_m1_pose = np.loadtxt('../top_down_m1_cam_pose.txt')
    cam_depth_scale = np.loadtxt('../camera_depth_scale.txt')
    pick_and_drop_m1(
        m1=m1, top_camera=cam,
        top_cam_m1_pose=top_cam_m1_pose,
        cam_depth_scale=cam_depth_scale)

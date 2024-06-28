import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import optimize
from real_world.m1 import M1
from real_world.camera.kinect import KinectClient
from real_world.setup import DEFAULT_ORN
from serial.tools import list_ports
measured_pts, observed_pts, observed_pix, world2camera = [None]*4


def calibrate(cam, m1, workspace_bounds, ee_to_checker=0.150, calib_grid_step=0.020):
    global measured_pts, observed_pts, observed_pix, world2camera
    # Constants

    checkerboard_offset = np.array(
        [0, 0, ee_to_checker])

    # Construct 3D calibration grid across workspace
    gridspace_x = np.linspace(
        workspace_bounds[0, 0],
        workspace_bounds[0, 1],
        1+int((workspace_bounds[0, 1]-workspace_bounds[0, 0])/calib_grid_step))
    gridspace_y = np.linspace(
        workspace_bounds[1, 0],
        workspace_bounds[1, 1],
        1+int((workspace_bounds[1, 1]-workspace_bounds[1, 0])/calib_grid_step))
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(
        gridspace_x, gridspace_y, workspace_bounds[2, 0]+ 0.1)
    num_calib_grid_pts = calib_grid_x.shape[0] * \
        calib_grid_x.shape[1]*calib_grid_x.shape[2]
    calib_grid_x.shape = (num_calib_grid_pts, 1)
    calib_grid_y.shape = (num_calib_grid_pts, 1)
    calib_grid_z.shape = (num_calib_grid_pts, 1)
    calib_grid_pts = np.concatenate(
        (calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

    # Move robot to each calibration point in workspace
    measured_pts = list()
    observed_pts = list()
    observed_pix = list()
    for calib_pt_idx in range(3):
        tool_position = calib_grid_pts[calib_pt_idx, :]
        tool_position[2] = workspace_bounds[2, 1]
        print(tool_position)
        m1.movel(use_pos=True,
                  params=list(tool_position) + DEFAULT_ORN,
                  blocking=True)
        time.sleep(1.0)

        while True:
            color_im, depth_im = cam.get_rgbd(repeats=10)
            chckr_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS +
                               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            bgr_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            gray_im = cv2.cvtColor(bgr_im, cv2.COLOR_RGB2GRAY)
            chckr_found, crnrs = cv2.findChessboardCorners(
                gray_im, chckr_size,
                None, 0
                # , None, cv2.CALIB_CB_ADAPTIVE_THRESH
            )
            if chckr_found:
                crnrs_refined = cv2.cornerSubPix(
                    gray_im, crnrs, (3, 3), (-1, -1), refine_criteria)
                block_pix = crnrs_refined[4, 0, :]
                break
            time.sleep(0.01)

        # Get observed checkerboard center 3D point in camera space
        block_z = depth_im[
            int(np.round(block_pix[1])),
            int(np.round(block_pix[0]))
        ]
        block_x = np.multiply(
            block_pix[1] - cam.color_intr[0, 2],
            block_z / cam.color_intr[0, 0]
        )
        block_y = np.multiply(
            block_pix[0] - cam.color_intr[1, 2],
            block_z / cam.color_intr[1, 1]
        )
        if block_z == 0:
            continue

        # Save calibration point and observed checkerboard center
        observed_pts.append([block_x, block_y, block_z])
        tool_position += checkerboard_offset
        measured_pts.append(tool_position)
        observed_pix.append(block_pix)

        # Draw and display the corners
        center = np.array(block_pix).astype(np.int16)
        vis_im = cv2.circle(
            color_im, center, 7, (0, 255, 0), 2)
        plt.imshow(vis_im)
        plt.show()
        # cv2.imshow('Calibration', cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(10)

    # Move robot back to home pose
    # m1.homej(blocking=True)

    measured_pts = np.asarray(measured_pts)
    observed_pts = np.asarray(observed_pts)
    observed_pix = np.asarray(observed_pix)
    world2camera = np.eye(4)

    # Estimate rigid transform with SVD (from Nghia Ho)
    def get_rigid_transform(A, B):
        assert len(A) == len(B)
        N = A.shape[0]  # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        # Dot is matrix multiplication for array
        H = np.dot(np.transpose(AA), BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def get_rigid_transform_error(z_scale):
        global measured_pts, observed_pts, observed_pix, world2camera

        # Apply z offset and compute new observed points
        # using camera intrinsics
        observed_z = observed_pts[:, 2:] * z_scale
        observed_x = np.multiply(
            observed_pix[:, [0]]-cam.color_intr[0, 2],
            observed_z/cam.color_intr[0, 0])
        observed_y = np.multiply(
            observed_pix[:, [1]]-cam.color_intr[1, 2],
            observed_z/cam.color_intr[1, 1])
        new_observed_pts = np.concatenate(
            (observed_x, observed_y, observed_z), axis=1)

        # Estimate rigid transform between measured points
        # and new observed points
        R, t = get_rigid_transform(np.asarray(
            measured_pts), np.asarray(new_observed_pts))
        t.shape = (3, 1)
        world2camera = np.concatenate(
            (np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(
            measured_pts)) + np.tile(t, (1, measured_pts.shape[0]))
        error = np.transpose(registered_pts) - new_observed_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error/measured_pts.shape[0])
        return rmse

    # Optimize z scale w.r.t. rigid transform error
    # 使用Nelder - Mead方法来优化z轴的缩放因子，以便使相机的深度测量值与实际深度尽可能匹配。
    print('Calibrating...')
    z_scale_init = 1
    optim_result = optimize.minimize(
        get_rigid_transform_error,
        np.asarray(z_scale_init),
        method='Nelder-Mead')
    camera_depth_offset = optim_result.x

    # Save camera optimized offset and camera pose
    print('Saving calibration files...')
    np.savetxt('camera_depth_scale.txt',
               camera_depth_offset,
               delimiter=' ')
    get_rigid_transform_error(camera_depth_offset)
    camera_pose = np.linalg.inv(world2camera)
    return camera_pose


if __name__ == "__main__":
    workspace_bounds = np.array([
        [0.240,  0.270],
        [-0.120, -0.05],
        [0.110, 0.115]
    ]) # x, y, z
    m1 = M1(list_ports.comports()[2].device, True)
    # m1.out_of_the_way()
    cam = KinectClient('127.0.0.1', 1111)
    np.savetxt('top_down_m1_cam_pose.txt',
               calibrate(cam, m1, workspace_bounds),
               delimiter=' ')


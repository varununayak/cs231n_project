'''
evaluate_odometry.py

Tools to evaluate KITTI odometry predictions

@author Varun Nayak
@date June 2020

'''
import numpy as np
from utils import load_poses

lengths = [100,200,300,400,500,600,700,800]

def trajectoryDistances(poses):
    dist = [0]
    for i in range(1, len(poses)):
        P1 = poses[i-1]
        P2 = poses[i]
        dx = P1[0,3] - P2[0,3]
        dy = P1[1,3] - P2[1,3]
        dz = P1[2,3] - P2[2,3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2 + dz**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, l):
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + l):
            return i
    return -1

def rotationError(pose_error):
    a = pose_error[0,0]
    b = pose_error[1,1]
    c = pose_error[2,2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(pose_error):
    dx = pose_error[0,3]
    dy = pose_error[1,3]
    dz = pose_error[2,3]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def calcSequenceErrors(poses_gt, poses_result):
    step_size = 10
    dist = trajectoryDistances(poses_gt);
    err = []
    for first_frame in range(0, len(poses_gt), step_size):
        for l in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, l)
            if (last_frame == -1):
                continue
            pose_delta_gt = np.matmul(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
            pose_delta_result = np.matmul(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
            pose_error = np.matmul(np.linalg.inv(pose_delta_result), pose_delta_gt)
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            num_frames = last_frame - first_frame + 1
            speed = l / (0.1 * num_frames)
            err.append((first_frame, r_err/l, t_err/l, l, speed))
    return err
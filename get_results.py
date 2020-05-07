#!/usr/bin/python3
"""
Script to load ground truth and predicted pose sequence 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

def load_poses(pose_path):
    # Read and parse the poses
    poses = []
    try:
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))  # to complete the homogenous representation
                poses.append(T_w_cam0)

    except:
        print('Error in finding or parsing filename: {}'.format(pose_path))

    return np.asarray(poses)


def plot_trajectories(ground_truth_poses, predicted_poses):
    x_g, z_g = ground_truth_poses[:,0,3], ground_truth_poses[:,2,3]
    x_p, z_p = predicted_poses[:,0,3], predicted_poses[:,2,3]
    plt.plot(x_g, z_g,'b-')
    plt.plot(x_p, z_p,'c--')
    plt.plot(x_g[0], z_g[0], 'ro')
    plt.plot(x_g[-1], z_g[-1], 'go')
    plt.grid()
    plt.legend(["Truth", "Pred", "Start", "Finish"])
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("z-coordinate [m]")
    plt.title("KITTI odometry ground truth vs predicted x-z coordinates")
    plt.show()
    # TODO(vn): Add more meaningful plots (as subplots)
    # TODO(vn): save plot somewhere

def compute_error_stats(ground_truth_poses, predicted_poses):
    x_g, z_g = ground_truth_poses[:,0,3], ground_truth_poses[:,2,3]
    x_p, z_p = predicted_poses[:,0,3], predicted_poses[:,2,3]
    x_mae = np.mean(np.abs(x_g - x_p)) 
    z_mae = np.mean(np.abs(z_g - z_p)) 
    print("X Mean Absolute Error: \t", x_mae)
    print("Z Mean Absolute Error: \t", z_mae)
    # TODO(vn): Add more meaningful statistics

def main():
    # Argparse ground truth pose and predicted csv file locations
    parser = argparse.ArgumentParser(description="Compare ground truth and predicted trajectories.")
    parser.add_argument("--ground_truth", type=str, nargs=1, default=['ground_truth_odometry/data_odometry_poses/dataset/poses/00.txt'])
    parser.add_argument("--predicted", type=str, nargs=1, default=['ground_truth_odometry/data_odometry_poses/dataset/poses/00.txt'])
    parsed_args = parser.parse_args()
    ground_truth_file_path, predicted_file_path = parsed_args.ground_truth[0], parsed_args.predicted[0]
    # Get poses
    ground_truth_poses = load_poses(ground_truth_file_path)
    predicted_poses = load_poses(predicted_file_path)
    assert len(ground_truth_poses) == len(predicted_poses), "Length mismatch between ground truth and predicted poses files"
    # print error statistics x,y,z mse, orientation mse
    compute_error_stats(ground_truth_poses, predicted_poses)
    # plot both on same graph # x-y,
    plot_trajectories(ground_truth_poses, predicted_poses)
    

if __name__=="__main__":
    main()
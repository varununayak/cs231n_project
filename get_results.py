#!/usr/bin/python3
"""
Script to load ground truth and predicted pose sequence
This script is still useless as of now 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from utils import load_poses

def get_xyz(poses):
    return poses[:,0], poses[:,1], poses[:,2]

def plot_trajectories(ground_truth_poses, predicted_poses):
    x_g, _, z_g = get_xyz(ground_truth_poses)
    x_p, _, z_p = get_xyz(predicted_poses)
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
    x_g, y_g, z_g = get_xyz(ground_truth_poses)
    x_p, y_p, z_p = get_xyz(predicted_poses)
    x_mae = np.mean(np.abs(x_g - x_p)) 
    z_mae = np.mean(np.abs(z_g - z_p))
    y_mae = np.mean(np.abs(y_g - y_p))
    print("---------------ERROR METRICS----------------") 
    print("X Mean Absolute Error: \t", x_mae)
    print("Z Mean Absolute Error: \t", z_mae)
    print("Y Mean Absolute Error: \t", y_mae)
    # TODO(vn): Add more meaningful statistics, those adapted by KITTI
    print("--------------------------------------------")

def main():
    # Argparse ground truth pose and predicted csv file locations
    parser = argparse.ArgumentParser(description="Compare ground truth and predicted trajectories.")
    parser.add_argument("--sequence", type=str, nargs=1, default=['01'])
    parsed_args = parser.parse_args()
    sequence = parsed_args.sequence[0]
    ground_truth_file_path = 'ground_truth_odometry/{}.txt'.format(sequence)
    predicted_file_path = 'predicted_odometry/{}.txt'.format(sequence)
    # Get poses
    ground_truth_poses = load_poses(ground_truth_file_path, get_only_translation=True)
    predicted_poses = load_poses(predicted_file_path, get_only_translation=True)
    if (len(ground_truth_poses) != len(predicted_poses)):
        mismatch = len(ground_truth_poses) - len(predicted_poses)
        print("Length mismatch of {} between ground truth and predicted poses files, fixing with zeros..".format(mismatch))
        zeros_pad = np.zeros((mismatch, ground_truth_poses.shape[1]))
        predicted_poses = np.vstack((zeros_pad, predicted_poses))
    # print error statistics x,y,z mse, orientation mse
    compute_error_stats(ground_truth_poses, predicted_poses)
    # plot both on same graph # x-y,
    plot_trajectories(ground_truth_poses, predicted_poses)
    

if __name__=="__main__":
    main()
"""
Script to load ground truth pose and display as a plot
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file."""
    pose_file = os.path.join(pose_path,'00.txt')
    print(pose_file)
    # Read and parse the poses
    poses = []
    try:
        with open(pose_file, 'r') as f:
            print("dounf")
            lines = f.readlines()
            
            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))  # to complete the homogenous representation
                poses.append(T_w_cam0)

    except:
        print('Error in finding or parsing file!')

    return poses


def main():
    pose_path = 'dataset/data_odometry_poses/dataset/poses'
    poses = load_poses(pose_path)
    first = 0
    last = len(poses) - 1
    x = []
    y = []
    for pose in poses:
        x.append(pose[0,3])
        y.append(pose[1,3])
    plt.plot(x,y)
    plt.plot(x[first], y[first], 'ro')
    plt.plot(x[last], y[last], 'go')
    plt.legend(["Path", "Start", "Finish"])
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")
    plt.title("KITTI odometry ground truth x-y coordinates")
    plt.show()

if __name__=="__main__":
    main()
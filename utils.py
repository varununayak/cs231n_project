'''
utils.py
'''
import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
from config import *
import tensorflow as tf

def mat_to_zyx_euler_angles(R):
    beta   = -np.arctan2(R[2,0], np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0]))
    c_beta = np.cos(beta)
    if abs(c_beta) < 1e-5: # Singularity
        alpha = 0
        R[1,1] = max(-1, min(1, R[1,1]))
        gamma = -sign(np.sin(beta)) * np.arccos(R[1,1])
    else:
        alpha  = np.arctan2(R[1,0]/c_beta, R[0,0]/c_beta)
        gamma  = np.arctan2(R[2,1]/c_beta, R[2,2]/c_beta)
    return alpha, beta, gamma

def load_poses(pose_path, get_only_translation=False):
    # Read and parse the poses
    poses = []
    try:
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))  # to complete the homogenous representation
                if get_only_translation:
                    T_w_cam0_xyz = np.asarray([T_w_cam0[0,3], T_w_cam0[1,3], T_w_cam0[2,3]])
                    poses.append(T_w_cam0_xyz) 
                else:
                    alpha, beta, gamma = mat_to_zyx_euler_angles(T[0:3,0:3])
                    T_w_cam0_xyz_abg = np.asarray([T_w_cam0[0,3], T_w_cam0[1,3], T_w_cam0[2,3], alpha, beta, gamma])
                    poses.append(T_w_cam0_xyz_abg)

    except:
        print('Error in finding or parsing filename: {}'.format(pose_path))
        exit(0)

    return np.asarray(poses)

def load_images(sequence='01'):
    filelist = glob.glob('../dataset/sequences/{}/image_2/*.png'.format(sequence))
    images = []
    
    ####### OPTION 1 #########
    for path in filelist:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img / 127.5 - 1.0)
    ##########################

    ####### OPTION 2 #########
    # images_raw = [cv.imread(fname) for fname in filelist]
    # images = []
    # for image in images_raw:
    #     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    #     image = np.divide(np.asarray(image), 127.5) - 1.0
    #     images.append(image)
    # images = np.asarray(images, dtype=np.float32)
    ##########################

    return images

def cumulate_poses(poses_predicted, init_pose):
    poses_predicted_cum = [init_pose]
    for pose_diff in poses_predicted:
        poses_predicted_cum.append(poses_predicted_cum[-1] + pose_diff)
    return np.asarray(poses_predicted_cum)

def plot_predictions_vs_truth(poses_predicted, poses_original):
    plt.plot(poses_original[:,0], poses_original[:,2])
    plt.plot(poses_predicted[:,0], poses_predicted[:,2])
    plt.show()

def preprocess_data(poses, images, use_absolute_pose_val):
    # First check some stuff for consistency
    N, dim_poses = poses.shape
    assert N == len(images), "Number of images and number of poses don't match!"
    assert dim_poses == DIM_PREDICTIONS, "Dimension mismatch between pose data and network output, check loaded data!"
    # For storage
    num_train = N - WINDOW_SIZE + 1
    poses_original = np.copy(poses[-num_train:]) # store for later
    # Store initial pose for cumulation
    init_pose = poses[-num_train-1]
    # TS Gen
    if (not use_absolute_pose_val):
        poses = poses  - np.vstack((np.zeros((1, dim_poses)), poses[:-1]))
    data_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(images, poses, WINDOW_SIZE, batch_size=64)
    return data_gen, poses_original, init_pose

def write_pose_to_file(poses, save_path):
    #answer = input("Do you want to save this predicted path to file? y/n: ")
    #if answer == "n":
    #   return
    N, dims = poses.shape
    if dims == 3:
        poses = np.hstack((np.ones((N,1)), np.zeros((N,2)), np.reshape(poses[:,0], (N,1)), 
                            np.zeros((N,1)), np.ones((N,1)), np.zeros((N,1)), np.reshape(poses[:,1], (N,1)),
                            np.zeros((N,2)), np.ones((N,1)), np.reshape(poses[:,2], (N,1))))
        print("Saving to {}".format(save_path))
        np.savetxt(save_path, poses, delimiter=" ")
    else: 
        print("Saving with dims != 3 not supported yet")




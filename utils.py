'''
utils.py
'''
import numpy as np
import glob
import matplotlib.pyplot as plt
from config import *
import tensorflow as tf
from threading import Thread, Lock
from PIL import Image
from rotations import mat_to_zyx_euler_angles, zyx_euler_angles_to_mat

mutex = Lock()

def load_poses(pose_path, get_only_translation=True):
    '''
    Returns a numpy array of poses stored in a file in pose_path
    according to the KITTI format. We convert the rotation matrix to 
    euler angles to reduce the redundancy of the representation
    for machine learning purposes.
    Args:
        pose_path: Path to the .txt file
        get_only_translation: return a N X 3 array if true else N X 6 (3 translation + 3 euler angles)
    Returns:
        poses: a N X 3 (or N X 6) numpy array of poses 
    '''
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
                    alpha, beta, gamma = mat_to_zyx_euler_angles(T_w_cam0[0:3,0:3]) # convert rotation matrix to euler angles
                    T_w_cam0_xyz_abg = np.asarray([T_w_cam0[0,3], T_w_cam0[1,3], T_w_cam0[2,3], alpha, beta, gamma])
                    poses.append(T_w_cam0_xyz_abg)
    except:
        print('Error in finding or parsing filename: {}'.format(pose_path))
        exit(0)
    return np.asarray(poses)

def load_images(sequence, model_name):
    '''
    Returns a python list of images from the dataset
    Args:
        sequence: String representing the sequence number e.g. '01'
        model_name: String representing the model name e.g. 'pyflownet'
    Returns:
        images: a python list of "images", definition of "image" depends on the model.
    '''
    images = []
    if (model_name == 'flowdispnet'):
        filepath_d = '../disparity_maps/{}/*.npy'.format(sequence)
        filepath_f = '../flow_dataset/{}/*.npy'.format(sequence)
        filelist_d = glob.glob(filepath_d)
        filelist_f = glob.glob(filepath_f)
        filelist_d.sort()
        filelist_f.sort()
        # Disparity maps
        images_d = []
        for path in filelist_d:
            imgs = np.load(path)
            for img in np.squeeze(imgs,axis=1):
                img = Image.fromarray(img).resize(size=(IMG_SIZE, IMG_SIZE)) 
                img = (np.asarray(img) - 80.0)/100.0    # Normalization to approx [-1, 1]
                images_d.append(np.expand_dims(img, axis=-1))
        # Optical Flow 
        images_f = []
        for path in filelist_f:
            img = np.load(path)
            img /= 80.0
            images_f.append(img)
        for img_f, img_d in zip(images_f, images_d[1:]):
            img = np.concatenate((img_f, img_d), axis=-1)
            images.append(img)
    else:
        if model_name == 'pyflownet' or model_name == 'rflownet':
            filepath = '../flow_dataset/{}/*.npy'.format(sequence)
        else:
            filepath = '../dataset/sequences/{}/image_2/*.png'.format(sequence)
        filelist = glob.glob(filepath)
        filelist.sort()
        for path in filelist:
            if model_name == 'pyflownet' or model_name == 'rflownet':
                img = np.load(path)
                images.append(img)
            else:
                img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
                img = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img / 127.5 - 1.0)
    return images

def cumulate_poses(poses_differential, init_pose):
    '''
    Cumulate the differential poses
    Args:
        poses_differential: N X 3(6) array of differential poses
        init_pose: starting pose
    Returns:
        N+1 X 3(6) array of cumulated poses
    '''
    return np.cumsum(np.concatenate((np.expand_dims(init_pose,axis=0), poses_differential), axis=0), axis=0)

def plot_trajectories(ground_truth_poses, predicted_poses):
    '''
    Plots and displays all meaningful trajectories ground truth vs prediction
    Args:
        ground_truth_poses: numpy array of ground truth poses
        predicted_poses: numpy array of predicted poses
    '''
    def get_xyz(poses):
        return poses[:,0], poses[:,1], poses[:,2]
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

def load_dataset(poses, images, total_num_list, poses_original_list, 
                init_pose_list, data_gen_list, model_name):
    '''
    Loads the images into a tensorflow Dataset object according to the model requirements 
    and appends it to the shared list (this is a thread worker, one worker per sequence)
    Args:
        poses: numpy array of target pose values (differential as per our model)
        images: list of images as per the definition of "image"
        *_list: shared list or set to populate, protected by a mutex to preserve order
        model_name: name of the model
    '''        
    # First check some stuff for consistency
    N, dim_poses = poses.shape
    assert dim_poses == DIM_PREDICTIONS, "Dimension mismatch between pose data and network output, check loaded data!"
    # For storage
    num_poses = (N - 1) if (model_name == 'pyflownet') else (N - WINDOW_SIZE + 1)
    # Store initial pose and original poses for later
    poses_original = np.copy(poses[-num_poses:])
    init_pose = poses[-num_poses-1]
    # Process poses to deltas
    poses = poses - np.vstack((np.zeros((1, dim_poses)), poses[:-1]))
    if model_name == 'pyflownet' or model_name == 'flowdispnet':
        data_gen = tf.data.Dataset.from_tensor_slices((images[:-1], poses[2:]))
    elif model_name == 'rflownet':
        data_gen = tf.keras.preprocessing.timeseries_dataset_from_array(images, poses[1:], WINDOW_SIZE).unbatch()
    else:
        print("Invalid model name")
        exit(0)
    mutex.acquire()
    data_gen_list.append(data_gen)
    poses_original_list.append(poses_original)
    init_pose_list.append(init_pose)
    total_num_list.append(num_poses)
    mutex.release()

def preprocess_data(poses_list, images_list, model_name, mode):
    '''
    Takes in a list of poses (one element per sequence) and images 
    and processes all of it into tensorflow dataset objects according 
    to the mode
    Args:
        poses_list: list of poses, each is a numpy array
        images_list: list of image sequences
        model_name: name of the model being used
        mode: 'train' or 'predict'
    Returns:
        if the mode is 'train'
            data_gen_train: tf dataset object of all shuffled training data
            dat_gen_val: tf dataset object of all shuffled validation data
        if the mode is 'predict'
            data_gen_list: list of data gen tf objects 
            poses_original_list: list of unmodified differential list of poses
            init_pose_list: list of initial poses for each sequence

    '''
    total_num_list, poses_original_list, init_pose_list, data_gen_list = [], [], [], []
    threads = []
    for poses, images in zip(poses_list, images_list):
        t = Thread(target=load_dataset, args=(poses, images, total_num_list, poses_original_list, init_pose_list, data_gen_list, model_name))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if mode == 'train':
        total_num = sum(total_num_list)
        data_gen = data_gen_list[0]
        for i in range(1, len(data_gen_list)):
            data_gen = data_gen.concatenate(data_gen_list[i])
        # Train - Val Split
        data_gen = data_gen.shuffle(total_num)
        num_train = int(0.8 * total_num)
        data_gen_train = data_gen.take(num_train)
        data_gen_val = data_gen.skip(num_train)
        data_gen_train = data_gen_train.batch(32)
        data_gen_val = data_gen_val.batch(1)
        return data_gen_train, data_gen_val
    elif mode == 'predict':
        return data_gen_list, poses_original_list, init_pose_list

def write_pose_to_file(poses, save_path):
    '''
    Writes the numpy array of poses to a .txt file in the KITTI format
    i.e. transformation matrix without the last row 
    '''
    N, dims = poses.shape
    if dims == 3:
        poses = np.hstack((np.ones((N,1)), np.zeros((N,2)), np.reshape(poses[:,0], (N,1)), 
                            np.zeros((N,1)), np.ones((N,1)), np.zeros((N,1)), np.reshape(poses[:,1], (N,1)),
                            np.zeros((N,2)), np.ones((N,1)), np.reshape(poses[:,2], (N,1))))
        print("Saving to {}".format(save_path))
        np.savetxt(save_path, poses, delimiter=" ")
    elif dims == 6:
        R = []
        for pose in poses: 
            R.append(zyx_euler_angles_to_mat(pose[3], pose[4], pose[5]))
        R = np.asarray(R)
        poses = np.hstack((
            np.reshape(R[:,0,0],(N,1)),     np.reshape(R[:,0,1],(N,1)),     np.reshape(R[:,0,2],(N,1)),      np.reshape(poses[:,0],(N,1)), 
            np.reshape(R[:,1,0],(N,1)),     np.reshape(R[:,1,1],(N,1)),     np.reshape(R[:,1,2],(N,1)),      np.reshape(poses[:,1],(N,1)),
            np.reshape(R[:,2,0],(N,1)),     np.reshape(R[:,2,1],(N,1)),     np.reshape(R[:,2,2],(N,1)),      np.reshape(poses[:,2],(N,1))
            ))
        print("Saving to {}".format(save_path))
        np.savetxt(save_path, poses, delimiter=" ")
    else:
        print("Error, couldn't save for dims = {}".format(dims))
'''
utils.py
'''
import numpy as np
import glob
import cv2 as cv

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

    return np.asarray(poses)

def load_images(img_size, sequence='01'):
    filelist = glob.glob('small_dataset/odometry/{}/image_2/*.png'.format(sequence))
    images_raw = [cv.imread(fname) for fname in filelist]
    images = []
    for i, image in enumerate(images_raw):
        image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_AREA)
        image = np.divide(np.asarray(image), 255.0)
        images.append(image)
    images = np.asarray(images)
    print("Images shape: ", images.shape)
    return images


def cumulate_poses(poses_predicted, init_pose):
    poses_predicted_cum = [init_pose]
    for pose_diff in poses_predicted:
        poses_predicted_cum.append(poses_predicted_cum[-1] + pose_diff)
    return np.asarray(poses_predicted_cum)



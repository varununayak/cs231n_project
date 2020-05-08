'''
utils.py
'''
import numpy as np

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
                    poses.append(T_w_cam0)

    except:
        print('Error in finding or parsing filename: {}'.format(pose_path))

    return np.asarray(poses)

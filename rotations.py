'''
rotations.py

Utility functions for rotation representation conversions

@author: Varun Nayak
@data: June 2020
'''

import numpy as np


def rot_x(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the x axis.
    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [1,             0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])


def rot_y(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the y axis.
    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rot_z(theta):
    """
    Returns a matrix that rotates vectors by theta radians about the z axis.
    Args:
        theta (float): angle to rotate by in radians
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [            0,              0, 1]
    ])

def mat_to_zyx_euler_angles(R):
    '''
    Convert a 3 X 3 rotation matrix to euler angles
    Args:
        R: a 3X3 rotation matrix
    Returns:
        alpha, beta, gamma: Euler angles (all floats)
    '''
    beta   = -np.arctan2(R[2,0], np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0]))
    c_beta = np.cos(beta)
    if abs(c_beta) < 1e-5: # Singularity
        alpha = 0
        R[1,1] = max(-1, min(1, R[1,1]))
        gamma = -np.sign(np.sin(beta)) * np.arccos(R[1,1])
    else:
        alpha  = np.arctan2(R[1,0]/c_beta, R[0,0]/c_beta)
        gamma  = np.arctan2(R[2,1]/c_beta, R[2,2]/c_beta)
    return alpha, beta, gamma

def zyx_euler_angles_to_mat(alpha, beta, gamma):
    """
    Converts ZYX Euler angles (rotation about z, then resulting y, then resulting x)
    into a rotation matrix.
    Args:
        alpha (float): angle in radians to rotate about z
        beta  (float): angle in radians to rotate about y
        gamma (float): angle in radians to rotate about x
    Returns:
        3x3 numpy array that is a rotation matrix
    """
    return rot_z(alpha).dot(rot_y(beta)).dot(rot_x(gamma))
#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
from utils import load_poses, load_images, cumulate_poses, plot_predictions_vs_truth, create_windowed_images, preprocess_data, write_pose_to_file
import os
import argparse
from config import *
from model import RCNN
import time

TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
#TRAIN_SEQUENCES = ['01', '01'] # for local machine
TEST_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
#TEST_SEQUENCES = ['01'] # for local
NUM_TRAIN_PASSES = 50

def main():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument("--mode", type=str, nargs=1, default=['train_and_predict'])
    parser.add_argument('--using_absolute_pose_val', dest='using_absolute_pose_val', action='store_true')
    parser.set_defaults(using_absolute_pose_val=False)
    parsed_args = parser.parse_args()
    mode, using_absolute_pose_val = parsed_args.mode[0], parsed_args.using_absolute_pose_val
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    print(f"-----------mode: {mode}, using_absolute_pose_val: {using_absolute_pose_val} -----------")
    print("-------------------------------------------------------------------------------------")
    poses_set = []
    images_set = []
    if (mode == 'predict_only'):
        sequences = TEST_SEQUENCES
        num_passes = 1
    else:
        sequences = TRAIN_SEQUENCES
        num_passes = NUM_TRAIN_PASSES
    for sequence in sequences:
        # Load ground truth
        poses_set.append(load_poses(f'ground_truth_odometry/{sequence}.txt', get_only_translation=True))
        # Load images (this call also resizes the image)
        images_set.append(load_images(sequence=sequence))
        print(f"Loaded Sequence {sequence}")
    for passno in range(num_passes):
        for j, sequence in enumerate(sequences):
            print(f"{mode} on sequence {sequence}, pass number {passno}")
            poses = poses_set[j]          
            images = images_set[j]          
            # Process images, poses
            data_gen, poses_original, init_pose = preprocess_data(poses, images, using_absolute_pose_val)
            # Create model from pretrained CNN 
            rcnn_model = RCNN()                                     
            # Train
            if (mode != 'predict_only'):
                rcnn_model.model.summary()
                rcnn_model.train(data_gen)
                if (mode =='train_only'):
                    continue
            # Predict on images
            poses_predicted = rcnn_model.predict(data_gen[0][0])
            for k in range(1, len(data_gen)):
                poses_predicted = np.vstack((poses_predicted, rcnn_model.predict(data_gen[k][0])))
            if (not using_absolute_pose_val):
                poses_predicted = cumulate_poses(poses_predicted, init_pose)
            plot_predictions_vs_truth(poses_predicted, poses_original)
            save_path = f"predicted_odometry/{sequence}.txt"
            write_pose_to_file(poses_predicted, save_path)
        pass
    return

if __name__=="__main__":
    main()

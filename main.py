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

#TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
TRAIN_SEQUENCES = ['01','01']
NUM_PASSES = 50

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
    for i in range(NUM_PASSES):
        for sequence in TRAIN_SEQUENCES:
            print(f"Training on sequence {sequence}, pass number {i}")
            # Load ground truth
            poses = load_poses(f'ground_truth_odometry/{sequence}.txt', get_only_translation=True)
            # Load images (this call also resizes the image)
            images = load_images(sequence=sequence) 
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
            for i in range(1, len(data_gen)):
                poses_predicted = np.vstack((poses_predicted, rcnn_model.predict(data_gen[i][0])))
            if (not using_absolute_pose_val):
                poses_predicted = cumulate_poses(poses_predicted, init_pose)
            plot_predictions_vs_truth(poses_predicted, poses_original)
            save_path = f"predicted_odometry/{sequence}.txt"
            write_pose_to_file(poses_predicted, save_path)
        pass
    return

if __name__=="__main__":
    main()

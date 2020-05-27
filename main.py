#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
from utils import load_poses, load_images, cumulate_poses, plot_predictions_vs_truth, preprocess_data, write_pose_to_file
import os
import argparse
from config import *
from model import Model
import time

#TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
TRAIN_SEQUENCES = ['01'] # for local machine
#TEST_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
TEST_SEQUENCES = ['01'] # for local
NUM_TRAIN_PASSES = 1

def parse_arguments():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument("--mode", type=str, nargs=1, default=['train_only'])
    parser.add_argument('--using_absolute_pose_val', dest='using_absolute_pose_val', action='store_true')
    parser.add_argument('--use_flow', dest='use_flow', action='store_true')
    parser.set_defaults(using_absolute_pose_val=False)
    parser.set_defaults(use_flow=False)
    parsed_args = parser.parse_args()
    mode, using_absolute_pose_val, use_flow = parsed_args.mode[0], parsed_args.using_absolute_pose_val, parsed_args.use_flow
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    print(f"------mode: {mode}, using_absolute_pose_val: {using_absolute_pose_val}, use_flow: {use_flow} ----------")
    print("-------------------------------------------------------------------------------------")
    return mode, using_absolute_pose_val, use_flow

def main():
    mode, using_absolute_pose_val, use_flow = parse_arguments()
    poses_set = []
    images_set = []
    # Populate sequences and num passes based on mode
    if (mode == 'predict_only'):
        sequences = TEST_SEQUENCES
        num_passes = 1
    else:
        sequences = TRAIN_SEQUENCES
        num_passes = NUM_TRAIN_PASSES
    # Load dataset 
    for sequence in sequences:
        # Load ground truth
        poses_set.append(load_poses(f'ground_truth_odometry/{sequence}.txt', get_only_translation=True))
        # Load images (this call also resizes the image)
        images_set.append(load_images(sequence, use_flow))
        print(f"Loaded Sequence {sequence}")
    for passnumber in range(num_passes):
        for seqidx, sequence in enumerate(sequences):
            print(f"{mode} on sequence {sequence}, pass number {passnumber}")
            poses = poses_set[seqidx]          
            images = images_set[seqidx]          
            # Process images, poses
            data_gen, poses_original, init_pose = preprocess_data(poses, images, using_absolute_pose_val, use_flow)
            # Create model from pretrained CNN 
            if use_flow:
                model_name = 'pyflownet'
            else:
                model_name = 'rflownetlite1.0'
            my_model = Model(model_name)                                     
            # Train
            if (mode != 'predict_only'):
                my_model.model.summary()
                my_model.train(data_gen)
                my_model.plot_history()
                if (mode =='train_only'):
                    continue
            # Predict on images
            if use_flow:
                poses_predicted = my_model.predict(np.asarray(images))
            else:
                poses_predicted = my_model.predict(data_gen[0][0])
                for k in range(1, len(data_gen)):
                    poses_predicted = np.vstack((poses_predicted, my_model.predict(data_gen[k][0])))
            if (not using_absolute_pose_val):
                poses_predicted = cumulate_poses(poses_predicted, init_pose)
            plot_predictions_vs_truth(poses_predicted, poses_original)
            save_path = f"predicted_odometry/{sequence}.txt"
            write_pose_to_file(poses_predicted, save_path)
        pass
    return

if __name__=="__main__":
    main()

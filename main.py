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
TRAIN_SEQUENCES = ['01', '01'] # for local machine
#TEST_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
TEST_SEQUENCES = ['01'] # for local

def parse_arguments():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument("--mode", type=str, nargs=1, default=['train'])
    parser.add_argument('--use_flow', dest='use_flow', action='store_true')
    parser.set_defaults(use_flow=False)
    parsed_args = parser.parse_args()
    mode, use_flow = parsed_args.mode[0], parsed_args.use_flow
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    print(f"----------------------mode: {mode},  use_flow: {use_flow} -----------------------")
    print("-------------------------------------------------------------------------------------")
    return mode, use_flow

def main():
    mode, use_flow = parse_arguments()
    poses_set, images_set = [], []
    # Populate sequences and num passes based on mode
    sequences = TRAIN_SEQUENCES if mode =='train' else TEST_SEQUENCES
    # Load dataset 
    for i, sequence in enumerate(sequences):
        # Load ground truth
        poses_set.append(load_poses(f'ground_truth_odometry/{sequence}.txt', get_only_translation=True))
        images_set.append(load_images(sequence, use_flow))
        print(f"Loaded Sequence {sequence}")
    # Process images, poses
    data_gen, poses_original_set, init_poses_set = preprocess_data(poses_set, images_set, use_flow)
    # Create model from pretrained CNN 
    model_name = 'pyflownet' if use_flow else 'rflownetlite1.0'
    my_model = Model(model_name)                                     
    # Train
    if (mode == 'train'):
        my_model.model.summary()
        my_model.train(data_gen)
        my_model.plot_history()
    # Predict on images
    elif (mode == 'predict'):
        for poses_original, init_pose, images, sequence in zip(poses_original_set, init_poses_set, images_set, sequences):
            if use_flow:
                poses_predicted = my_model.predict(np.asarray(images))
            else:
                poses_predicted = my_model.predict(data_gen[0][0])
                for k in range(1, len(data_gen)):
                    poses_predicted = np.vstack((poses_predicted, my_model.predict(data_gen[k][0])))
            poses_predicted = cumulate_poses(poses_predicted, init_pose)
            plot_predictions_vs_truth(poses_predicted, poses_original)
            write_pose_to_file(poses_predicted, save_path=f"predicted_odometry/{sequence}.txt")
    else:
        print(f"INVALID MODE {mode}")
    return

if __name__=="__main__":
    main()

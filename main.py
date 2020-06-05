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
from threading import Thread, Lock
import gc

#TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
TRAIN_SEQUENCES = ['01'] # for local machine
TEST_SEQUENCES = ['01'] # one at a time

mutex = Lock()

def parse_arguments():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument("--mode", type=str, nargs=1, default=['train'])
    parser.add_argument("--model_name", type=str, nargs=1, default=['pyflownet'])
    parsed_args = parser.parse_args()
    mode, model_name = parsed_args.mode[0], parsed_args.model_name[0]
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    print(f"----------------------mode: {mode},  model_name: {model_name} -----------------------")
    print("-------------------------------------------------------------------------------------")
    return mode, model_name

def load_data(get_only_translation, sequence, model_name, poses_set, images_set):
    poses = load_poses(f'ground_truth_odometry/{sequence}.txt', get_only_translation=False)
    images = load_images(sequence, model_name)
    mutex.acquire()
    poses_set.append(poses)
    images_set.append(images)
    mutex.release()
    print(f"Loaded Sequence {sequence}")

def main():
    mode, model_name = parse_arguments()
    poses_set, images_set = [], []
    # Populate sequences and num passes based on mode
    sequences = TRAIN_SEQUENCES if mode =='train' else TEST_SEQUENCES
    # Load dataset
    threads = []
    for sequence in sequences:
        # Load ground truth
        t = Thread(target=load_data, args=(True, sequence, model_name, poses_set, images_set))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    # Create model from pretrained CNN 
    my_model = Model(model_name)                                     
    # Train
    if (mode == 'train'):
        # Process images, poses
        data_gen_train, data_gen_val = preprocess_data(poses_set, images_set, model_name, mode)
        del images_set, poses_set # Memory conservation
        my_model.model.summary()
        gc.collect()
        my_model.train(data_gen_train, data_gen_val)
        my_model.plot_history()
    # Predict on images
    elif (mode == 'predict'):
        # Process images, poses
        data_gen_list, poses_original_set, init_poses_set = preprocess_data(poses_set, images_set, model_name, mode)
        del images_set # Memory conservation
        gc.collect()
        # For each sequence
        for data_gen, poses_original, init_pose, sequence in zip(data_gen_list, poses_original_set, 
                                                                        init_poses_set, sequences):
            poses_predicted = my_model.predict(data_gen.batch(1))
            poses_predicted = cumulate_poses(poses_predicted, init_pose)
            plot_predictions_vs_truth(poses_predicted, poses_original)
            write_pose_to_file(poses_predicted, save_path=f"predicted_odometry/{sequence}.txt")
    else:
        print(f"INVALID MODE {mode}")
    return

if __name__=="__main__":
    main()

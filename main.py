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

def main():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument('--predict_only', dest='predict_only', action='store_true')
    parser.add_argument('--using_absolute_pose_val', dest='using_absolute_pose_val', action='store_true')
    parser.add_argument("--sequence", type=str, nargs=1, default=['01'])
    parser.set_defaults(predict_only=False)
    parser.set_defaults(using_absolute_pose_val=False)
    parsed_args = parser.parse_args()
    predict_only, using_absolute_pose_val, sequence = parsed_args.predict_only, parsed_args.using_absolute_pose_val, parsed_args.sequence[0]
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    # Load ground truth
    poses = load_poses('ground_truth_odometry/{}.txt'.format(sequence), get_only_translation=True)
    # Load images (this call also resizes the image)
    images = load_images(sequence=sequence) 
    # Process images, poses
    poses, poses_original, images_windowed, init_pose = preprocess_data(poses, images, using_absolute_pose_val)
    # Create model from pretrained CNN 
    rcnn_model = RCNN()                                     
    if (not predict_only):
        rcnn_model.model.summary()
        rcnn_model.train(images_windowed, poses)
    poses_predicted = rcnn_model.predict(images_windowed)
    if (not using_absolute_pose_val):
        poses_predicted = cumulate_poses(poses_predicted, init_pose)
    plot_predictions_vs_truth(poses_predicted, poses_original)
    save_path = "predicted_odometry/"+sequence+".txt"
    write_pose_to_file(poses_predicted, save_path)

if __name__=="__main__":
    main()

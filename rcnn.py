#!/usr/bin/python3
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from utils import load_poses

IMG_SIZE = 128 #224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
WINDOW_SIZE = 5 #10
WINDOW_IMG_SHAPE = (WINDOW_SIZE, IMG_SIZE, IMG_SIZE, 3)
DIM_PREDICTIONS = 3 # 3 for translation, 3 more probably (euler angles) for translation + rotation

def main():
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    # Load ground truth
    poses = load_poses('ground_truth_odometry/data_odometry_poses/dataset/poses/01.txt', get_only_translation=True)
    # Load images
    filelist = glob.glob('small_dataset/odometry/01/image_2/*.png')
    images = [cv.imread(fname) for fname in filelist]
    for i, image in enumerate(images):
        image = cv.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA)
        image = np.divide(np.asarray(image), 255.0)
        images[i] = image
    images = np.asarray(images)
    print("Images shape: ", images.shape)
    num_train, dim_poses = poses.shape
    assert num_train == images.shape[0], "Number of images and number of poses don't match!"
    assert dim_poses == DIM_PREDICTIONS, "Dimension mismatch between pose data and network output, check loaded data!"
    num_train = num_train - WINDOW_SIZE + 1
    poses = poses[-num_train:]
    # Create windowed dataset of images
    images_windowed = np.zeros((num_train, WINDOW_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(num_train):
        for j in range(WINDOW_SIZE):
            images_windowed[i,j] = np.copy(images[i+j])
    print("Windowed Images shape", images_windowed.shape)
    print("Poses shape ", poses.shape)
    # Create model from pretrained CNN 
    cnn_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3),
                                                weights='imagenet')
    cnn_model = tf.keras.models.Sequential([cnn_model, tf.keras.layers.Flatten()])  
    cnn_model_time_dist = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=WINDOW_IMG_SHAPE)], name='time_dist_cnn_model')                                        
    cnn_model_time_dist.trainable = False
    print("Output shape of cnn_model_time_dist: ", cnn_model_time_dist.layers[-1].output_shape)
    rnn_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(16, return_sequences=True, name='lstm1'),
                        tf.keras.layers.LSTM(16, return_sequences=False, name='lstm2'),
                        tf.keras.layers.Dense(64, activation="relu"),
                        tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
    model = tf.keras.models.Sequential([cnn_model_time_dist, rnn_model])
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(images_windowed, poses, batch_size = 16, epochs = 10)
    poses_predicted = model.predict(images_windowed)
    print("Poses shape:", poses.shape)
    print("Poses Pred shape:", poses_predicted.shape)
    plt.plot(poses[:,0], poses[:,2])
    plt.plot(poses_predicted[:,0], poses_predicted[:,2])
    plt.show()



if __name__=="__main__":
    main()

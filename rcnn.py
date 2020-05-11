#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
import matplotlib.pyplot as plt
from utils import load_poses, load_images, cumulate_poses
import os
import argparse

IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
WINDOW_SIZE = 5
WINDOW_IMG_SHAPE = (WINDOW_SIZE, IMG_SIZE, IMG_SIZE, 3)
DIM_PREDICTIONS = 3 # 3 for translation, 3 more probably (euler angles) for translation + rotation

def create_time_dist_cnn_model(use_pretrained=True):
    if use_pretrained:
        cnn_model = tf.keras.applications.ResNet50V2(include_top=False, pooling='max', input_shape=(IMG_SIZE,IMG_SIZE,3),
                                                weights='imagenet')
        cnn_model = tf.keras.models.Sequential([cnn_model, tf.keras.layers.Flatten()])
        cnn_model.trainable = False
    else:
        cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), (1,1), 'same', activation='relu',data_format='channels_last'),
                        tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
                        tf.keras.layers.MaxPool2D(),
                        tf.keras.layers.Conv2D(32, (3,3), (1,1), 'same', activation='relu',data_format='channels_last'),
                        tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
                        tf.keras.layers.MaxPool2D(),
                        tf.keras.layers.Conv2D(64, (5,5), (1,1), 'same', activation='relu', data_format='channels_last'),
                        tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
                        tf.keras.layers.SpatialDropout2D(rate=0.3),
                        tf.keras.layers.MaxPool2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(2048, activation='relu')])
    cnn_model_time_dist = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=WINDOW_IMG_SHAPE)], name='time_dist_cnn_model')                                        
    return cnn_model_time_dist

def create_rnn_model():
    rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, name='lstm_1')),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False, name='lstm_2')),
                        tf.keras.layers.Dense(64, activation="relu"),
                        tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
    return rnn_model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae')<0.02):
            print("\nReached 0.02 mae so cancelling training!")
            self.model.stop_training = True

def predict_using_model(model, images_windowed, poses_original, init_pose):
    N = images_windowed.shape[0]
    poses_predicted = model.predict(images_windowed[0:50])
    for i in range(50,N,50):
        lower = i
        upper = min(i + 50, N)
        poses_predicted = np.vstack((poses_predicted,model.predict(images_windowed[lower:upper])))
    print("Poses shape:", poses_original.shape)
    print("Poses Pred shape:", poses_predicted.shape)
    poses_predicted_cum = cumulate_poses(poses_predicted, init_pose)
    plt.plot(poses_original[:,0], poses_original[:,2])
    plt.plot(poses_predicted_cum[:,0], poses_predicted_cum[:,2])
    plt.show()

def train_model(model, images_windowed, poses, save_weights_callback):
    N = images_windowed.shape[0]
    model.fit(images_windowed, poses, batch_size = 8, epochs = 20, callbacks=[save_weights_callback, myCallback()])
    # for i in range(0,N,100):
    #     lower = i
    #     upper = min(i + 100, N)
    #     model.fit(images_windowed[lower:upper], poses[lower:upper], batch_size = 8, epochs = 20, callbacks=[save_weights_callback, myCallback()])
    # pass

def load_saved_weights(model, use_pretrained):
    if use_pretrained:
        checkpoint_path = "training_1/cp.ckpt"
    else:
        checkpoint_path = "training_0/cp.ckpt"
    # Create a callback that saves the model's weights
    save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    try:
        model.load_weights(checkpoint_path)   
        print("Loaded previously saved weights")
    except:
        print("Could not load weights from file")
    return save_weights_callback

def main():
    parser = argparse.ArgumentParser(description="RCNN")
    parser.add_argument('--predict_only', dest='predict_only', action='store_true')
    parser.set_defaults(predict_only=False)
    predict_only = parser.parse_args().predict_only
    print("----------------------- Using TensorFlow version:", tf.__version__,"---------------------------")
    # Load ground truth
    poses = load_poses('ground_truth_odometry/data_odometry_poses/dataset/poses/01.txt', get_only_translation=True)
    # Load and preprocess images
    images = load_images(img_size=IMG_SIZE)
    # Check some stuff for consistency
    num_train, dim_poses = poses.shape
    assert num_train == images.shape[0], "Number of images and number of poses don't match!"
    assert dim_poses == DIM_PREDICTIONS, "Dimension mismatch between pose data and network output, check loaded data!"
    # num train after windowing
    num_train = num_train - WINDOW_SIZE + 1
    # Process poses current - previous
    poses_original = np.copy(poses[-num_train:]) # store for later
    init_pose = poses[-num_train-1]
    poses = poses[-num_train:] - poses[-num_train-1:-1]
    # Create windowed dataset of images
    images_windowed = np.zeros((num_train, WINDOW_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(num_train):
        for j in range(WINDOW_SIZE):
            images_windowed[i,j] = np.copy(images[i+j])
    print("Windowed Images shape: {}, Poses shape: {}".format(images_windowed.shape, poses.shape))
    # Create model from pretrained CNN 
    use_pretrained = True
    cnn_model_time_dist = create_time_dist_cnn_model(use_pretrained=use_pretrained)
    print("Output shape of cnn_model_time_dist: ", cnn_model_time_dist.layers[-1].output_shape)
    rnn_model = create_rnn_model()
    model = tf.keras.models.Sequential([cnn_model_time_dist, rnn_model])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    save_weights_callback = load_saved_weights(model, use_pretrained)                                         
    if (not predict_only):
        model.summary()
        train_model(model, images_windowed, poses, save_weights_callback)
    predict_using_model(model, images_windowed, poses_original, init_pose)



if __name__=="__main__":
    main()

'''
rcnn architecure
'''
from config import * 
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

NUM_EPOCHS = 100

class Model(object):

    def __init__(self, model_name):
        # Checkpoint to save and load weights from
        self.checkpoint_path = "{}/cp.ckpt".format(model_name)
        # Store model name
        self.model_name = model_name
        # Flag
        self._loaded_weights = False
        # Initialize the model
        self._init_model()
    
    def _init_model(self):
        # Create the model
        ## ------------------------------------rflownetlite1.0-----------------------------------##
        if self.model_name == 'rflownetlite1.0':
            cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(4, (5,5), (2,2), 'same', activation=None,data_format='channels_last'),
                                tf.keras.layers.Conv2D(4, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(4, (3,3), (1,1), 'same'),
                                tf.keras.layers.Conv2D(8, (3,3), (1,1), 'same'),
                                tf.keras.layers.Conv2D(16, (3,3), (1,1), 'same'),
                                tf.keras.layers.Flatten()])
            ## ------------------------------------rnn model-------------------------------------##
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, name='lstm_1')),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False, name='lstm_2')),
                            tf.keras.layers.Dense(16, activation="selu"),
                            tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
            model = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=WINDOW_IMG_SHAPE)], name='time_dist_cnn_model')
            model = tf.keras.models.Sequential([model, rnn_model])
        ## ------------------------------------pyflownet-----------------------------------##
        elif self.model_name == 'pyflownet':
            model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (7,7), (2,2), 'same', input_shape=(IMG_SIZE, IMG_SIZE, 2)),
                                tf.keras.layers.Conv2D(32, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (1,1), 'same'),
                                tf.keras.layers.Conv2D(64, (1,1), (1,1), 'same'),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(256, activation="selu"),
                                tf.keras.layers.Dense(DIM_PREDICTIONS)], name='pyflownet')
        ## ------------------------------------rflownet-----------------------------------##
        elif self.model_name == 'rflownet':
            cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (5,5), (2,2), 'same', input_shape=(IMG_SIZE, IMG_SIZE, 2)),
                                tf.keras.layers.Conv2D(32, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (1,1), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (1,1), 'same'),
                                tf.keras.layers.Flatten()])
            ## ------------------------------------rnn model-------------------------------------##
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, return_sequences=True, name='lstm_1'),
                            tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_2'),
                            tf.keras.layers.Dense(128, activation="selu"),
                            tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
            model = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=(None, IMG_SIZE, IMG_SIZE, 2))], name='time_dist_cnn_model')
            model = tf.keras.models.Sequential([model, rnn_model])
        elif self.model_name == 'rdispnet':
            cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (5,5), (2,2), 'same', input_shape=(DISP_H, DISP_W, 1)),
                                tf.keras.layers.Conv2D(32, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (5,5), (2,2), 'same'),
                                tf.keras.layers.Conv2D(64, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (2,2), 'same'),
                                tf.keras.layers.Conv2D(128, (3,3), (1,1), 'same'),
                                tf.keras.layers.Conv2D(64, (1,1), (1,1), 'same'),
                                tf.keras.layers.Flatten()])
            ## ------------------------------------rnn model-------------------------------------##
            rnn_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, return_sequences=True, name='lstm_1'),
                            tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_2'),
                            tf.keras.layers.Dense(128, activation="selu"),
                            tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
            model = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=(None, DISP_H, DISP_W, 1))], name='time_dist_cnn_model')
            model = tf.keras.models.Sequential([model, rnn_model])
        else:
            sys.exit("Model name {} is invalid.".format(self.model_name))
        self.model = model
        # History
        self._history = None
    
    def train(self, data_gen_train, data_gen_val):
        save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1)
        # Define optimizer and compile model
        if (self.model_name == 'pyflownet'):
            opt = 'adam'
        else:
            opt = 'sgd'
        self.model.compile(loss='mae', optimizer=opt, metrics=['mae'])
        # Load any existing weights if they exist
        self._load_existing_weights()
        # Fit on the data
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'
        else:
            device = '/device:CPU:0'
        with tf.device(device):
            self._history = self.model.fit(data_gen_train, validation_data=data_gen_val, epochs=NUM_EPOCHS, callbacks=[save_weights_callback])
    
    def plot_history(self):
        loss = self._history.history['loss']
        val_loss = self._history.history['val_loss']
        plt.plot(range(len(loss)), loss,'r-')
        plt.plot(range(len(val_loss)), val_loss,'b--')
        plt.grid()
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.legend(['Train Loss', 'Validation Loss'])
        dateTimeObj = datetime.now()
        timestamp = str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "_" + str(dateTimeObj.hour) + ":" + str(dateTimeObj.minute)
        plt.savefig('images/loss_{}.png'.format(timestamp))
        plt.show()

    def predict(self, data_gen):
        if not self._loaded_weights:
            self._load_existing_weights()
        poses_predicted = self.model.predict(data_gen)
        return poses_predicted
    
    def _load_existing_weights(self):     # Try loading existing weights if they exist
        try:
            self.model.load_weights(self.checkpoint_path)   
            print("Loaded previously saved weights for model {}".format(self.model_name))
        except:
            print("Could not load weights from file!!!! Ignore if training afresh or new model.")
        self._loaded_weights = True

            


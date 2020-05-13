'''
rcnn architecure
'''
from config import * 
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
import sys
import numpy as np


class RCNN(object):

    def __init__(self, model_name='rflownetlite1.0'):
        # Create the model
        ## ------------------------------------rflownetlite.10-----------------------------------##
        if model_name == 'rflownetlite1.0':
            cnn_model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(4, (5,5), (1,1), 'same', activation=None,data_format='channels_last'),
                                tf.keras.layers.Conv2D(4, (3,3), (1,1), 'same', activation=None,data_format='channels_last'),
                                tf.keras.layers.Conv2D(4, (3,3), (1,1), 'same', activation=None, data_format='channels_last'),
                                tf.keras.layers.Conv2D(8, (3,3), (1,1), 'same', activation=None, data_format='channels_last'),
                                tf.keras.layers.Flatten()])
        ## ------------------------------------rresnet50v2---------------------------------------##
        elif model_name == 'rresnet50v2':
            cnn_model = tf.keras.applications.ResNet50V2(include_top=False, pooling='max', input_shape=(IMG_SIZE,IMG_SIZE,3), weights='imagenet')
            cnn_model = tf.keras.models.Sequential([cnn_model, tf.keras.layers.Flatten()])
            cnn_model.trainable = False
        else:
            sys.exit("Model name {} is invalid.".format(model_name))
        ## ------------------------------------common rnn model-------------------------------------##
        rnn_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, name='lstm_1')),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False, name='lstm_2')),
                        tf.keras.layers.Dense(16, activation="selu"),
                        tf.keras.layers.Dense(DIM_PREDICTIONS)], name='rnn_model')
        cnn_model_time_dist = tf.keras.models.Sequential([tf.keras.layers.TimeDistributed(cnn_model, input_shape=WINDOW_IMG_SHAPE)], name='time_dist_cnn_model')
        self.model = tf.keras.models.Sequential([cnn_model_time_dist, rnn_model])
        # Checkpoint to save and load weights from
        self.checkpoint_path = "{}/cp.ckpt".format(model_name)
        # Store model name
        self.model_name = model_name
        # Flag
        self._loaded_weights = False

    
    def train(self, images_windowed, poses):
        # Callbacks for stopping training and saving weights
        class mae_stop_callback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if(logs.get('mae')<0.01):
                    print("\nReached 0.01 mae so cancelling training!")
                    self.model.stop_training = True
        save_weights_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1)
        # Define optimizer and compile model
        opt = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        self.model.compile(loss='mae', optimizer=opt, metrics=['mae'])
        # Load any existing weights if they exist
        self._load_existing_weights()
        # Fit on the data
        self.model.fit(images_windowed, poses, batch_size = 64, epochs = 1, callbacks=[save_weights_callback, mae_stop_callback()])

    def predict(self, images_windowed):
        if not self._loaded_weights:
            self._load_existing_weights()
        N = images_windowed.shape[0]
        poses_predicted = self.model.predict(images_windowed[0:50])
        for i in range(50,N,50):
            lower, upper = i, min(i + 50, N)
            poses_predicted = np.vstack((poses_predicted, self.model.predict(images_windowed[lower:upper])))
        return poses_predicted
    
    def _load_existing_weights(self):     # Try loading existing weights if they exist
        try:
            self.model.load_weights(self.checkpoint_path)   
            print("Loaded previously saved weights for model {}".format(self.model_name))
        except:
            print("Could not load weights from file!!!! Ignore if training afresh or new model.")
        self._loaded_weights = True

            


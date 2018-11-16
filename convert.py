import argparse
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D,ZeroPadding3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import videoto3d
from tqdm import tqdm
import keras
from keras.models import model_from_json
import h5py


def scan_hdf52(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        elems = []
        for k, v in g.items():
            if isinstance(v, h5.Dataset):
                elems.append(v.name)
            elif isinstance(v, h5.Group) and recursive:
                elems.append((v.name, scan_node(v, tabs=tabs + tab_step)))
        return elems

    with h5py.File(path, 'r') as f:
        return scan_node(f)
scan_hdf52("caffe_weights/sports1M_weights.h5", recursive=True,tab_step=2)

#futures_data = h5['futures_data']  # VSTOXX futures data
#options_data = h5['options_data']  # VSTOXX call option data


'''
model = Sequential()
# 1st layer group
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(112,112,16,3), padding='same',
                 name='conv1', strides=(1, 1, 1)))
#  input_shape=(3, 16, 112, 112)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))

# 2nd layer group
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))

# 3rd layer group

model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3a', strides=(1, 1, 1)))
model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv3b', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))

# 4th layer group
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv4a', strides=(1, 1, 1)))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv4b', strides=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))

# 5th layer group
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv5a', strides=(1, 1, 1)))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv5b', strides=(1, 1, 1)))
model.add(ZeroPadding3D(padding=(0, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
model.add(Flatten())

# FC layers group
model.add(Dense(4096, activation='relu', name='fc6'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu', name='fc7'))
model.add(Dropout(.5))
model.add(Dense(487, activation='softmax', name='fc8'))
print(model.summary())

model.load_weights('caffe_weights/sports1M_weights.h5')
'''
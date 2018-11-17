import os
import numpy as np

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
import argparse
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D,ZeroPadding3D)

from keras.models import Sequential
import tensorflow as tf
import keras
import h5py

''' BACKEND must be TENSORFLOW
This is a script to convert Theano models (Theano Backend, TH dim ordering)
to the other possible backend / dim ordering combinations.
Given weights and model for TH-kernels-TH-dim-ordering, produces a folder with
- TH-kernels-TF-dim-ordering
- TF-kernels-TH-dim-ordering
- TF-kernels-TF-dim-ordering
Needs 3 important inputs:
1) Theano model (model with TH dim ordering)
2) Tensorflow model (model with TF dim ordering)
3) Weight file for Theano model (theano-kernels-th-dim-ordering)
Supports : Multiple weights for same model (auto converts different weights for same model)
Usage:
1) Place script in the same directory as the weight file directory. If you want to place somewhere
   else, then you must provide absolute path to the weight files below instead of relative paths.
2) Edit the script to create your model :
    a) Import your model building script above (in the imports section)
    b) Set `th_dim_model` = ... (create your th dim model here and set it to th_dim_model)
    c) Set `tf_dim_model` = ... (create your tf dim model here and set it to tf_dim_model)
    d) Add the path to the weight files in `model_weights`.
       Note : The weight files must be for the Theano model (theano kernels, th dim ordering)
3) Run the script.
4) Use the weight files in the created folders : ["tf-kernels-tf-dim/", "tf-kernels-th-dim/", "th-kernels-tf-dim/"]
'''
def getmodel_tf():
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(112,112,16,3), padding='same',
                     name='conv1', strides=(1, 1, 1)))
    #  input_shape=(3, 16, 112, 112)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))

    # 2nd layer group
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 2, 2), padding='valid', name='pool2'))

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
    return model

def getmodel_th():
    model_th = Sequential()
    # 1st layer group
    model_th.add(Conv3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112)))
    model_th.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model_th.add(Conv3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model_th.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model_th.add(Conv3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model_th.add(Conv3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model_th.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model_th.add(Conv3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model_th.add(Conv3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model_th.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model_th.add(Conv3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model_th.add(Conv3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model_th.add(ZeroPadding3D(padding=(0, 1, 1)))
    model_th.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model_th.add(Flatten())
    # FC layers group
    model_th.add(Dense(4096, activation='relu', name='fc6'))
    model_th.add(Dropout(.5))
    model_th.add(Dense(4096, activation='relu', name='fc7'))
    model_th.add(Dropout(.5))
    model_th.add(Dense(487, activation='softmax', name='fc8'))
    print(model_th.summary())
    return model_th


K.set_image_data_format('channels_first')
th_dim_model = getmodel_th() # Create your theano model here with TH dim ordering

K.set_image_data_format('channels_last')
tf_dim_model = getmodel_tf() # Create your tensorflow model with TF dimordering here

model_weights = '/home/mahnaz/____PycharmProjects/3DCNN/3DCNN/caffeweights/sports1M_weights.h5' # Add names of theano model weight file paths here.
                     # These weights are assumed to be for  theano backend
                     # (th kernels) with th dim ordering!
f = h5py.File(model_weights, 'r')
np.savetxt('datafile.txt', f['layer'][...])
print(list(f))
"""
No need to edit anything below this. Simply run the script now after
editing the above 3 inputs.
"""


def shuffle_rows(original_w, nb_last_conv, nb_rows_dense):
    ''' Note :
    This algorithm to shuffle dense layer rows was provided by Kent Sommers (@kentsommer)
    in a gist : https://gist.github.com/kentsommer/e872f65926f1a607b94c2b464a63d0d3
    '''
    converted_w = np.zeros(original_w.shape)
    count = 0
    for index in range(original_w.shape[0]):
        if (index % nb_last_conv) == 0 and index != 0:
            count += 1
        new_index = ((index % nb_last_conv) * nb_rows_dense) + count
        print("index from " + str(index) + " -> " + str(new_index))
        converted_w[index] = original_w[new_index]

    return converted_w


first_dense = True
nb_last_conv = 0

for dirpath in ["tf-kernels-channels-last-dim-ordering/", "tf-kernels-channels-first-dim-ordering/", "th-kernels-channels-last-dim-ordering/"]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Converts (theano kernels, th dim ordering) to (tensorflow kernels, th dim ordering)
K.set_image_dim_ordering('tf')
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)
    convert_all_kernels_in_model(th_dim_model)

    th_dim_model.save_weights("tf-kernels-channels-first-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-first-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (tensorflow kernels, tf dim ordering)
K.set_image_dim_ordering('th')
for weight_fn in model_weights:
    print(weight_fn)
    th_dim_model.load_weights(weight_fn) # th-kernels-th-dim
    convert_all_kernels_in_model(th_dim_model) # tf-kernels-th-dim

    count_dense = 0
    for layer in th_dim_model.layers:
        if layer.__class__.__name__ == "Dense":
            count_dense += 1

    if count_dense == 1:
        first_dense = False # If there is only 1 dense, no need to perform row shuffle in Dense layer

    print("Nb layers : ", len(th_dim_model.layers))

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = th_layer.get_weights() # tf-kernels-th-dim
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights) # tf-kernels-tf-dim

            nb_last_conv = th_layer.nb_filter # preserve last number of convolutions to use with dense layers
            print("Converted layer %d : %s" % (index + 1, th_layer.name))
        else:
            if th_layer.__class__.__name__ == "Dense" and first_dense:
                weights = th_layer.get_weights()
                nb_rows_dense_layer = weights[0].shape[0] // nb_last_conv

                print("Magic Number 1 : ", nb_last_conv)
                print("Magic nunber 2 : ", nb_rows_dense_layer)

                weights[0] = shuffle_rows(weights[0], nb_last_conv, nb_rows_dense_layer)
                tf_dim_model.layers[index].set_weights(weights)

                first_dense = False
                print("Shuffled Dense Weights layer and saved %d : %s" % (index + 1, th_layer.name))
            else:
                tf_dim_model.layers[index].set_weights(th_layer.get_weights())
                print("Saved layer %d : %s" % (index + 1, th_layer.name))


    tf_dim_model.save_weights("tf-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done tf-kernels-channels-last-dim-ordering %s" % weight_fn)


# Converts (theano kernels, th dim ordering) to (theano kernels, tf dim ordering)
for weight_fn in model_weights:
    th_dim_model.load_weights(weight_fn)

    for index, th_layer in enumerate(th_dim_model.layers):
        if th_layer.__class__.__name__ in ['Conv1D',
                                           'Conv2D',
                                           'Conv3D',
                                           'AtrousConvolution1D'
                                           'AtrousConvolution2D',
                                           'Conv2DTranspose',
                                           'SeparableConv2D',
                                           'DepthwiseConv2D',
                                           ]:
            weights = th_layer.get_weights()
            weights[0] = weights[0].transpose((2, 3, 1, 0))
            tf_dim_model.layers[index].set_weights(weights)
        else:
            tf_dim_model.layers[index].set_weights(th_layer.get_weights())

        print("Changed dim %d : %s" % (index + 1, th_layer.name))

    tf_dim_model.save_weights("th-kernels-channels-last-dim-ordering/%s" % weight_fn, overwrite=True)
    print("Done th-kernels-channels-last-dim-ordering %s" % weight_fn)



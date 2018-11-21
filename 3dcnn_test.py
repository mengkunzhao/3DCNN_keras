########################################################################################################################
#        Modified implementation of 3DCNN in Keras _ For gesture recognitions _ Calearn Isolated Gestures              #
########################################################################################################################

import argparse
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,MaxPooling3D,ZeroPadding3D, BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import videoto3d
from tqdm import tqdm
from keras import optimizers
from tensorflow.python.client import device_lib
import keras
from keras.models import model_from_json
import h5py


# Setting the Keras to use GPU Acceleration
print(device_lib.list_local_devices())
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 32} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Helper Function for plotting model accuracy and loss
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()

# Helper function for saving the model
def save_history(history, result_dir):
    loss = history['loss']
    acc = history['acc']
    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('loss\tacc\t\n')
        fp.write('{}\t{}\t\n'.format(loss, acc))


# Helper function to load data from video file
def loaddata(video_list, vid3d, nclass, result_dir, skip=True):
    dir = '/tank/gesrecog/chalearn/train/'
    output = open("Test_list_sorted.txt", 'w')
    test1ist = list(sorted(open(os.path.join(dir + video_list), 'r')))
    for line in sorted(test1ist, key=lambda line: int(line.split(' ')[2])):
        print(line)
        output.write(line)

    vid_dirs = open("Test_list_sorted.txt", 'r')
    X = []
    labels = []
    pbar = tqdm(total=len(vid_dirs))

    for rows in vid_dirs:
        pbar.update(1)
        name = os.path.join(dir, rows.split(' ')[0])
        temp = vid3d.video3d(name, skip=skip)
#Checking if the input video is broken or not
        if temp.shape[0] == 16:
            X.append(temp)
            label = rows.split(' ')[2]
            labels.append(label.split('\n')[0])
# The original labels start from one, but our system needs them to start from 0
    label = np.asarray(labels,dtype=int) - 1

    pbar.close()
    return np.array(X).transpose((0, 2, 3, 4, 1)), label


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101', help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=249)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)

    args = parser.parse_args()

#Initializing the dimentions of the frames
    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3
    nb_classes = args.nclass
    fname_npz = 'dataset_test_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)

#If the dataset is already stored in npz file:
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
#If not, we load the data with the helper function and save it for future use:
        x, y = loaddata(args.videos, vid3d, args.nclass, args.output, args.skip)
        Y= np_utils.to_categorical(y, nb_classes)
        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        X = X.astype('float32')
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved test dataset to dataset_test.npz.')

    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

# Define model
    model = model_from_json(open('3dcnn_500_32_adam.json', 'r').read())
    model.load_weights('3dcnn_500_32_adam.h5')
    model.summary()
    print("Loaded model from disk")

#List of Optimizers we used:
    adam = optimizers.Adam(lr=0.01, decay=0.0001, amsgrad=False)
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001, nesterov=True)
    ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

#Compiling and fitting the model
    model.compile(loss= 'categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

#Evaluating the model on the test_set
    score = model.evaluate(X, Y, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    save_history(score, args.output)

if __name__ == '__main__':
    main()

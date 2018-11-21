########################################################################################################################
#       Modified implementation of 3DCNN in Keras _ For Training gesture recognitions _ Calearn Isolated Gestures      #
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
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


# Helper function to load data from video file
def loaddata(video_list, vid3d, skip=True):
    dir = '/tank/gesrecog/chalearn/train/'
    output = open("Train_list_sorted.txt", 'w')
    train1ist = list(sorted(open(video_list, 'r')))
    for line in sorted(train1ist, key=lambda line: int(line.split(' ')[2])):
        print(line)
        output.write(line)

    vid_dirs = list(open("Train_list_sorted.txt", 'r'))
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
    fname_npz = 'dataset_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)

#If the dataset is already stored in npz file:
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
#If not, we load the data with the helper function and save it for future use:
        x, y = loaddata(args.videos, vid3d, args.skip)
        Y= np_utils.to_categorical(y, nb_classes)
        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        X = X.astype('float32')
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')

    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))


    X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42)


# Define model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        X.shape[1:]), padding="same"))
    model.add(LeakyReLU())
    model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()
    plot_model(model, show_shapes=True,
          to_file=os.path.join(args.output, 'model.png'))

#List of Optimizers we used:
    adam = optimizers.Adam(lr=0.01, decay=0.0001, amsgrad=False)
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001, nesterov=True)
    ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

#Compiling and fitting the model
    model.compile(loss= 'categorical_crossentropy',
                  optimizer=nadam, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)

#Saving the model
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output,'3dcnn_{}_{}_nadam.json'.format(args.epoch,args.batch)) , 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output,'3dcnn_{}_{}_nadam.h5'.format(args.epoch,args.batch)))

#Evaluation on test data if available
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)

if __name__ == '__main__':
    main()
import argparse
import os
import keras
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D,BatchNormalization)
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
from keras import optimizers

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import keras


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 32} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

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


def loaddata(vid_list, vid3d, nclass, result_dir, color=False, skip=True):
    files = os.listdir(vid_list)
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(vid_list, filename)
        label = vid3d.get_UCF_classname(filename)
        if label not in labellist:
            if len(labellist) >= nclass:
                continue
            labellist.append(label)
        labels.append(label)
        X.append(vid3d.video3d(name, color=color, skip=skip))

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=101)
    parser.add_argument('--output', type=str, required=True)
#    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)
    args = parser.parse_args()

    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3 #if args.color else 1
    #fname_npz = 'dataset_{}_{}_{}.npz'.format(
    #    args.nclass, args.depth, args.skip)

    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    #if os.path.exists(fname_npz):
    #    loadeddata = np.load(fname_npz)
    #    X, Y = loadeddata["X"], loadeddata["Y"]
    #    print(X.shape)
    #else:
    x, y = loaddata(args.videos, vid3d, args.nclass, args.output, args.color, args.skip)
    #    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    #    Y = np_utils.to_categorical(y, nb_classes)
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(X.shape[1:]), padding='same',
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
    model.add(Dense(nb_classes, activation='softmax', name='fc8'))
    print(model.summary())
    return model


    adam = optimizers.Adam(lr=0.01, decay=0.0001, amsgrad=False)
    sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True)
    ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    nadam = optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss= 'categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    model.summary()

#    fname_npz = 'dataset_chunk_{}.npz'.format(j)
#    loadeddata = np.load(fname_npz)
    #x, y = loaddata(args.videos, vid3d, args.nclass,
     #               args.output, args.color, args.skip)
    #X_, Y_ = loadeddata["X"], loadeddata["Y"]
    Y= np_utils.to_categorical(y, nb_classes)
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    X = X.astype('float32')

    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42)
    #    np.savez(fname_npz, X=X, Y=Y)
     #   print('Saved dataset to dataset.npz.')
    #print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    # Define model
    plot_model(model, show_shapes=True,
          to_file=os.path.join(args.output, 'model.png'))

   # X_train, X_test, Y_train, Y_test = train_test_split(
   #     X, Y, test_size=0.2, random_state=43)

    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)
    #model_json = model.to_json()
    #if not os.path.isdir(args.output):
     #   os.makedirs(args.output)
    #with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
     #   json_file.write(model_json)
    #model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    #plot_history(history, args.output)
    #save_history(history, args.output)


if __name__ == '__main__':
    main()

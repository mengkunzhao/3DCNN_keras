import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Input, Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, Input, average, ZeroPadding3D)
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras import optimizers
from tqdm import tqdm
import videoto3d


def plot_history(history, result_dir, name):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(name)))
    plt.close()


def save_history(history, result_dir, name):
    loss=history.history['loss']
    acc=history.history['acc']
    val_loss=history.history['val_loss']
    val_acc=history.history['val_acc']
    nb_epoch=len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(name)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loaddata(video_list, vid3d, nclass, result_dir, skip=True):
    dir = '/tank/gesrecog/chalearn/train/'
    vid_dirs = list(open(os.path.join(dir + video_list), 'r'))
    chunk_size = 500
    chunk_range = int(len(vid_dirs) / chunk_size)
    print(chunk_range)
    pbar = tqdm(total=len(vid_dirs))
    for i in range(chunk_range+1):
        X = []
        labels = []
        print(chunk_size*i, (i+1)*chunk_size-1)
        for rows in vid_dirs[i*chunk_size:(i+1)*chunk_size-1]:
            pbar.update(1)
            name = os.path.join(dir, rows.split(' ')[0])
            temp = vid3d.video3d(name, skip=skip)
            if temp.shape[0] == 16:
                X.append(temp)
                label = rows.split(' ')[2].split('\n')[0]
                labels.append(label.split('\n')[0])
        #print(labels)

        label = np.asarray(labels,dtype=int) -1
        print(len(label))
        fname_npz = 'dataset_chunk_{}.npz'.format(i)
        np.savez(fname_npz, X=np.array(X).transpose((0, 1, 4, 2, 3)), Y= label)
    pbar.close()
    print('loading data is done')
    #return np.array(X), labels
    # return np.array(X).transpose((0, 1, 4, 2, 3)), labels


def create_3dcnn(input_shape, nb_classes):
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(input_shape), padding='same',
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


def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=249)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--nmodel', type=int, default=3)

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    img_rows, img_cols, frames = 112, 112, args.depth
    channel = 3

    #vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    #fname_npz = 'dataset_{}_{}_{}.npz'.format(args.nclass, args.depth, args.skip)
    #loaddata(args.videos, vid3d, args.nclass,args.output, args.skip)
    #X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    #Y = np_utils.to_categorical(y, nb_classes)
    #    if os.path.exists(fname_npz):

    models=[]
    accuracy = []
    loss_ = []
    for j in range(72):
        fname_npz = 'dataset_chunk_{}.npz'.format(j)
        loadeddata = np.load(fname_npz)
        X_, Y_ = loadeddata["X"], loadeddata["Y"]
        Y= np_utils.to_categorical(Y_, nb_classes)
        X = X_.reshape((X_.shape[0], img_rows, img_cols, frames, channel))
        print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
        break
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=4)
        for i in range(args.nmodel):
            hist = []
            print('model{}:'.format(i))
            models.append(create_3dcnn((img_rows, img_cols, 16, 3), nb_classes))
            adam = optimizers.Adam(lr=0.001, decay=0.0001, amsgrad=False)
            models[-1].compile(loss='categorical_crossentropy',
                               optimizer=adam, metrics=['accuracy'])

    # Define model
            history_ = models[-1].fit(X_train, Y_train, validation_data=(
                X_test, Y_test), batch_size=args.batch, nb_epoch=args.epoch, verbose=1, shuffle=True)
            hist.append(history_)
            print(len(hist))
            print(history_.shape)
            plot_history(hist , args.output, '{}_{}'.format(j,i))
            save_history(hist, args.output, '{}_{}'.format(j, i))
            #accuracy.append(history.history(['acc']))
            #loss_.append(history.history['loss'])
            #val_loss_.append(history.history['val_loss'])
            #val_accuracy.append(history.history['val_acc'])
        #loss1 = sum(loss_)/len(loss_)
        #acc1 = sum(accuracy)/len(accuracy)
        #val_accuracy1 =  sum(val_accuracy)/len(val_accuracy)
        #val_loss1 = sum(val_loss_)/len(val_loss_)

            #plot_history(history , args.output, '{}_{}'.format(i,j))
            #save_history(history, args.output, '{}_{}'.format(i,j))

        model_inputs = [Input(shape=X.shape[1:]) for _ in range (args.nmodel)]
        model_outputs = [models[i](model_inputs[i]) for i in range (args.nmodel)]
        model_outputs = average(inputs=model_outputs)
        model = Model(inputs=model_inputs, outputs=model_outputs)
        model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
        model.summary()
        plot_model(model, show_shapes=True,
            to_file=os.path.join(args.output, 'model.png'))

   # model_json=model.to_json()
   # with open(os.path.join(args.output, 'Chalearn_3dcnnmodel.json'), 'w') as json_file:
   #     json_file.write(model_json)
   # model.save_weights(os.path.join(args.output, 'Chalearn_3dcnnmodel.hd5'))

        loss, acc=model.evaluate([X_test]*args.nmodel, Y_test, verbose=0)
        loss_.append(loss)
        accuracy.append(acc)
    acc1 = sum(accuracy)/len(accuracy)
    loss1 = sum(loss_)/len(loss_)
    with open(os.path.join(args.output, 'result.txt'), 'w') as f:
            f.write('Test loss: {}\nTest accuracy:{}'.format(loss1, acc1))

    print('merged model:')
    print('Test loss:', loss1)
    print('Test accuracy:', acc1)

if __name__ == '__main__':
    main()

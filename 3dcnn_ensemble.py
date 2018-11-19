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
    #files=os.listdir(vid_dirs)
    X=[]
    labels=[]
    labellist=[]
    temp_shape = []
    pbar=tqdm(total=len(vid_dirs))

    for rows in vid_dirs:
        pbar.update(1)
        name=os.path.join(dir, rows.split(' ')[0])
        print(name)
        #X.append(temp)
        #print(np.array(X).size)
        temp = vid3d.video3d(name, skip=skip)
        if temp.shape is (16,112,112,3):
            X.append(temp)

        #if toload.split('/')[-1] == rows.split(' ')[0].split('/')[-1]:
            if rows == '.DS_Store':
                continue
        #print(name)
            label=rows.split(' ')[2]
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
        #with open(('classes.txt'), 'w+') as ss:
        #    ss.write('{}, {} , {} , {} \n'.format(str(name) , str(checkframe), str(checkret) , str(temp_shape)))
        #ss.close()
    pbar.close()


    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i]=num
    return np.array(X) , labels
    #return np.array(X).transpose((0, 1, 4, 2, 3)), labels


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

    args=parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    img_rows, img_cols, frames=112, 112, args.depth
    channel=3

    vid3d=videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    fname_npz = 'dataset_{}_{}_{}.npz'.format(args.nclass, args.depth, args.skip)

#    if os.path.exists(fname_npz):
#    loadeddata = np.load(fname_npz)
#    X, Y = loadeddata["X"], loadeddata["Y"]
#        print(X.shape)
#    else:
    x, y = loaddata(args.videos, vid3d, args.nclass,args.output, args.skip)
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    Y = np_utils.to_categorical(y, nb_classes)

    X = X.astype('float32')
    np.savez(fname_npz, X=X, Y=Y)
    print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
'''
    X_train, X_test, Y_train, Y_test=train_test_split(
        X, Y, test_size=0.2, random_state=4)

    # Define model
    models=[]
    for i in range(args.nmodel):
        print('model{}:'.format(i))
        models.append(create_3dcnn(X.shape[1:], nb_classes))
        adam = optimizers.Adam(lr=0.001, decay=0.0001, amsgrad=False)

        models[-1].compile(loss='categorical_crossentropy',
                      optimizer= adam, metrics=['accuracy'])
        history = models[-1].fit(X_train, Y_train, validation_data=(
            X_test, Y_test), batch_size=args.batch, nb_epoch=args.epoch, verbose=1, shuffle=True)
        plot_history(history , args.output, i)
        save_history(history, args.output, i)

    model_inputs = [Input(shape=X.shape[1:]) for _ in range (args.nmodel)]
    model_outputs = [models[i](model_inputs[i]) for i in range (args.nmodel)]
    model_outputs = average(inputs=model_outputs)
    model = Model(inputs=model_inputs, outputs=model_outputs)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

    model.summary()
    plot_model(model, show_shapes=True,
         to_file=os.path.join(args.output, 'model.png'))

    model_json=model.to_json()
    with open(os.path.join(args.output, 'Chalearn_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'Chalearn_3dcnnmodel.hd5'))

    loss, acc=model.evaluate([X_test]*args.nmodel, Y_test, verbose=0)
    with open(os.path.join(args.output, 'result.txt'), 'w') as f:
        f.write('Test loss: {}\nTest accuracy:{}'.format(loss, acc))

    print('merged model:')
    print('Test loss:', loss)
    print('Test accuracy:', acc)

'''
if __name__ == '__main__':
    main()

import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,MaxPooling3D,ZeroPadding3D, BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU
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


def loaddata(video_list, vid3d, color , skip=True ):
    dir = '/tank/gesrecog/chalearn/'
    vid_dirs = list(open(os.path.join(dir,video_list), 'r'))
    #files=os.listdir(vid_dirs)
    print(len(vid_dirs))
    X = []
    labels = []
    pbar = tqdm(total=len(vid_dirs))

    for rows in vid_dirs:
        pbar.update(1)
        if color == False:
            name = os.path.join(dir, rows.split(' ')[1])
        else:
            name = os.path.join(dir, rows.split(' ')[0])
        temp = vid3d.video3d(name, color, skip=skip  )
        # Checking if the input video is broken or not
        if temp.shape[0] == 16:
            X.append(temp)
            #print(temp.shape)
            label = rows.split(' ')[2]
            # print(label)
            labels.append(label.split('\n')[0])
    print(labels)
    # The original labels start from one, but our system needs them to start from 0
    label_ = np.asarray(labels, dtype=int) - 1
    #print(label_)
    pbar.close()
    if color == True:
        return np.array(X).transpose((0, 2, 3, 4, 1)), label_
    else:
        return np.array(X).transpose((0, 2, 3, 1)), label_



def create_3dcnn(input_shape, nb_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
        input_shape), padding="same"))
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

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()
    return model

def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train', type=str, default='train.txt')
    parser.add_argument('--valid', type=str, default='valid.txt')
    parser.add_argument('--nclass', type=int, default=249)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--nmodel', type=int, default=3)

    args=parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    img_rows, img_cols, frames= 32, 32, args.depth
    channel_c = 3
    channel_d = 1
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    nb_classes = args.nclass
    fname_npz_train_c = 'dataset_trainc_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)
    fname_npz_train_d= 'dataset_traind_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)
    fname_npz_valid_c = 'dataset_validc_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)
    fname_npz_valid_d= 'dataset_validd_{}_{}_{}.npz'.format(
        args.nclass, args.depth, args.skip)


    if os.path.exists(fname_npz_valid_c):
        loadeddata = np.load(fname_npz_valid_c)
        Xvc, Yvc = loadeddata["X"], loadeddata["Y"]
    else:
    # If not, we load the data with the helper function and save it for future use:
        xvc, yvc = loaddata(args.valid, vid3d, color = True, skip =True)
        Yvc = np_utils.to_categorical(yvc, nb_classes)
        Xvc = xvc.reshape((xvc.shape[0], img_rows, img_cols, frames, channel_c))
        Xvc = Xvc.astype('float32')
        np.savez(fname_npz_valid_c, X=Xvc, Y=Yvc)
        print('Saved valid dataset to dataset_train.npz.')

    if os.path.exists(fname_npz_valid_d):
        loadeddata = np.load(fname_npz_valid_d)
        Xvd, Yvd = loadeddata["X"], loadeddata["Y"]
    else:
    # If not, we load the data with the helper function and save it for future use:
        xvd, yvd = loaddata(args.valid, vid3d,color = False, skip =True)
        Yvd = Yvc
        Xvd = xvd.reshape((xvd.shape[0], img_rows, img_cols, frames, channel_d))
        Xvd = Xvd.astype('float32')
        np.savez(fname_npz_valid_d, X=Xvd, Y=Yvd)
        print('Saved valid dataset to dataset_train.npz.')

    if os.path.exists(fname_npz_train_c):
        loadeddata = np.load(fname_npz_train_c)
        Xtc, Ytc = loadeddata["X"], loadeddata["Y"]
    else:
    # If not, we load the data with the helper function and save it for future use:
        xtc, ytc = loaddata(args.valid, vid3d, color = True, skip =True)
        Ytc = np_utils.to_categorical(ytc, nb_classes)
        Xtc = xtc.reshape((xtc.shape[0], img_rows, img_cols, frames, channel_c))
        Xtc = Xtc.astype('float32')
        np.savez(fname_npz_train_c, X=Xtc, Y=Ytc)
        print('Saved valid dataset to dataset_train.npz.')

    if os.path.exists(fname_npz_train_d):
        loadeddata = np.load(fname_npz_train_d)
        Xtd, Ytd = loadeddata["X"], loadeddata["Y"]
    else:
    # If not, we load the data with the helper function and save it for future use:
        xtd, ytd = loaddata(args.valid, vid3d,color = False,  )
        Ytd = Ytc
        Xtd = xtd.reshape((xtd.shape[0], img_rows, img_cols, frames, channel_d))
        Xtd = Xtd.astype('float32')
        np.savez(fname_npz_train_d, X=Xtd, Y=Ytd)
        print('Saved valid dataset to dataset_train.npz.')


    X_train_c, X_test_c, Y_train_c, Y_test_c= Xtc, Xvc, Ytc, Yvc
    X_train_d, X_test_d, Y_train_d, Y_test_d= Xtd, Xvd, Ytd, Yvd

    # Define model
    models=[]
    models.append(create_3dcnn(Xtc.shape[1:], nb_classes))
    adam = optimizers.Adam(lr=0.01, decay=0.0001, amsgrad=False)
    models[-1].compile(loss='categorical_crossentropy',
                       optimizer=adam, metrics=['accuracy'])
    history1 = models[-1].fit(X_train_c, Y_train_c, validation_data=(
        X_test_c, Y_test_c), batch_size=args.batch, nb_epoch=args.epoch, verbose=1, shuffle=True)

    models.append(create_3dcnn(Xtd.shape[1:], nb_classes))
    adam = optimizers.Adam(lr=0.01, decay=0.0001, amsgrad=False)
    models[-1].compile(loss='categorical_crossentropy',
                       optimizer=adam, metrics=['accuracy'])
    history2 = models[-1].fit(X_train_d, Y_train_d, validation_data=(
        X_test_d, Y_test_d), batch_size=args.batch, nb_epoch=args.epoch, verbose=1, shuffle=True)


    model_inputs = [Input(shape=Xtd.shape[1:]),Input(shape=Xtc.shape[1:])]


    model_outputs = [models[0](model_inputs[0]), models[1](model_inputs[1])]
    model_outputs = average(inputs=model_outputs)
    model = Model(inputs=model_inputs, outputs=model_outputs)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

    model.summary()
    #plot_model(model, show_shapes=True,
    #     to_file=os.path.join(args.output, 'model.png'))

    #model_json_c=models[0].to_json()
    #with open(os.path.join(args.output, 'Chalearn_3dcnnmodel_c.json'), 'w') as json_file:
    #    json_file.write(model_json_c)
    #models[0].save_weights(os.path.join(args.output, 'Chalearn_3dcnnmodel_c.hd5'))


    #model_json_c=models[0].to_json()
    #with open(os.path.join(args.output, 'Chalearn_3dcnnmodel_c.json'), 'w') as json_file:
    #    json_file.write(model_json_c)
    #models[0].save_weights(os.path.join(args.output, 'Chalearn_3dcnnmodel_c.hd5'))

'''
    loss, acc=model.evaluate([X_test]*args.nmodel, Y_test, verbose=0)
    with open(os.path.join(args.output, 'result.txt'), 'w') as f:
        f.write('Test loss: {}\nTest accuracy:{}'.format(loss, acc))

    print('merged model:')
    print('Test loss:', loss)
    print('Test accuracy:', acc)
'''

if __name__ == '__main__':
    main()

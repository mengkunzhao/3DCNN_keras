import argparse
import os

import matplotlib
matplotlib.use('AGG')
<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
=======
import numpy as np

>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a
from keras.models import model_from_json

import videoto3d
from tqdm import tqdm

def loaddata_test(video_dir, vid3d, color=False, skip=True):
    files = os.listdir(video_dir)
    X = []
<<<<<<< HEAD
    labellist = []
=======
>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a
    pbar = tqdm(total=len(files))
    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        X.append(vid3d.video3d(name, color=color, skip=skip))
    pbar.close()
    if color:
<<<<<<< HEAD
        return np.array(X).transpose((0, 2, 3, 4, 1))
    else:
        return np.array(X).transpose((0, 2, 3, 1)) 
=======
        return np.array(X).transpose((0, 2, 3, 4, 1)) #, labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)) #, labels
>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a

def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
<<<<<<< HEAD
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--model_dir', type=str, required=True)
=======
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--output', type=str, required=True)
>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()

    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3 if args.color else 1
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
<<<<<<< HEAD
    x = loaddata_test(args.videos, vid3d, args.color, args.skip)
=======
    x, y = loaddata_test(args.videos, vid3d, args.color, args.skip)
>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a
    X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
    X = X.astype('float32')

# load json and create model
    json_file = open('ucf101_3dcnnmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("ucf101_3dcnnmodel.hd5")
    print("Loaded model from disk")
    y_pred = loaded_model.predict_classes(X)
<<<<<<< HEAD
    print("Predicted=%s" % y_pred)
   
if __name__ == '__main__':
    main()


=======
    print("Predicted=%s" % y_pred))

if __name__ == '__main__':
    main()

>>>>>>> a52b20b0df797c1eb77990a8f388a7efa104661a

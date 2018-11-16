import argparse
import os

import matplotlib
matplotlib.use('AGG')
import numpy as np

from keras.models import model_from_json

import videoto3d
from tqdm import tqdm

def loaddata_test(video_dir, vid3d, color=False, skip=True):
    files = os.listdir(video_dir)
    X = []
    pbar = tqdm(total=len(files))
    for filename in files:
        pbar.update(1)
        if filename == '.DS_Store':
            continue
        name = os.path.join(video_dir, filename)
        X.append(vid3d.video3d(name, color=color, skip=skip))
    pbar.close()
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)) #, labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)) #, labels

def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=10)
    args = parser.parse_args()

    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3 if args.color else 1
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)
    x, y = loaddata_test(args.videos, vid3d, args.color, args.skip)
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
    print("Predicted=%s" % y_pred))

if __name__ == '__main__':
    main()


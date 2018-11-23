# 3DCNN fir Gesture recognition
 Inplementation of 3D Convolutional Neural Network for video classification using [Keras](https://keras.io/)(with [tensorflow](https://www.tensorflow.org/) as backend).

## Options
Options of 3dcnn.py are as following:  
`--batch`   batch size, default is 128  
`--epoch`   the number of epochs, default is 100  
`--videos`  a name of directory where dataset is stored, default is UCF101  
`--nclass`  the number of classes you want to use, default is 101  
`--output`  a directory where the results described above will be saved  
`--color`   use RGB image or grayscale image, default is False  
`--skip`    get frames at interval or contenuously, default is True  
`--depth`   the number of frames to use, default is 10  

Options of 3dcnn\_ensemble.py are almost same as those of 3dcnn.py.
You can use `--nmodel` option to set the number of models.

Options of visualize\_input.py are as follows:  
`--model` saved json file of a model  
`--weights` saved hd5 file of a model weights  
`--layernames` True to show layer names of a model, default is False  
`--name` the name of a layer which will be maximized  
`--index` the index of a layer output which will be maximized  
`--iter` the number of iteration, default is 20  

You can see more information by using `--help` option
## Demo
You can execute like the following:
```sh
python 3dcnn.py --batch 32 --epoch 50 --videos dataset/ --nclass 10 --output 3dcnnresult/ --color True --skip False --depth 15
```


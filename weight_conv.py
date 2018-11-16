import caffe.proto.caffe_pb2 as caffe
import numpy as np

p = caffe.NetParameter()
p.ParseFromString(
    open('caffe_weights/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read()
)

def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                W[i, j, k] = np.rot90(W[i, j, k], 2)
    return W

params = []
conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
fc_layers_indx = [22, 25, 28]

for i in conv_layers_indx:
    layer = p.layers[i]
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
        layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
        layer.blobs[0].height, layer.blobs[0].width
    )
    weights_p = rot90(weights_p)
    params.append([weights_p, weights_b])
for i in fc_layers_indx:
    layer = p.layers[i]
    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
        layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
        layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T
    params.append([weights_p, weights_b])
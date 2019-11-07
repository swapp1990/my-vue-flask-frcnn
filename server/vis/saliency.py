from __future__ import absolute_import
import numpy as np
from scipy.ndimage.interpolation import zoom
import sys
sys.path.append("..")

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K

from vis.optimizer import Optimizer
from vis.losses import ActivationMaximization
from util import viz_utils
from backend import tensorflow_backend as tf_b

def visualize_saliency(model, layer_idx, class_indices, seed_input):
    """Generates an attention heatmap over the `seed_input` for maximizing `filter_indices`
    output in the given `layer_idx`.
    """
    model = tf_b.modify_model_backprop(model, 'guided')

    losses = [
        (ActivationMaximization(model.layers[layer_idx], class_indices), -1)
    ]

    input_tensor = model.input
    opt = Optimizer(input_tensor, losses)
    grads = opt.minimize(seed_input=seed_input, max_iter=1)[1]

    keepdims = False
    if not keepdims:
        channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
        grads = np.max(grads, axis=channel_idx)
    return viz_utils.normalize(grads)[0]

def visualize_cam(model, layer_idx, class_indices, seed_input):
    model = tf_b.modify_model_backprop(model, 'guided')
    penultimate_layer = _find_penultimate_layer(model, layer_idx)
    # `ActivationMaximization` outputs negative gradient values for increase in activations. Multiply with -1
    # so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], class_indices), -1)
    ]

    input_tensor = model.input
    penultimate_output = penultimate_layer.output

    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output)
    _, grads, penultimate_output_value = opt.minimize(seed_input=seed_input, max_iter=1)
    grads = grads / (np.max(grads) + K.epsilon())
    #print(grads.shape) #(1,7,7,512) - grads for all 512 features are outputted for the penultimate layer

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    output_dims = viz_utils.get_img_shape(penultimate_output_value)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    heatmap = np.maximum(heatmap, 0)
    input_dims = viz_utils.get_img_shape(input_tensor)[2:]
    # Figure out the zoom factor.
    zoom_factor = [i / (j * 1.0) for i, j in iter(zip(input_dims, output_dims))]
    heatmap = zoom(heatmap, zoom_factor)
    return viz_utils.normalize(heatmap)

def _find_penultimate_layer(model, layer_idx):
    penultimate_layer_idx = None
    for idx, layer in viz_utils.reverse_enumerate(model.layers[:layer_idx - 1]):
        if isinstance(layer, Wrapper):
            #print(idx, layer, layer.layer)
            layer = layer.layer
        if isinstance(layer, (_Conv, _Pooling1D, _Pooling2D, _Pooling3D)):
            penultimate_layer_idx = idx
            break
    
    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Conv` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))
    
    return model.layers[penultimate_layer_idx]
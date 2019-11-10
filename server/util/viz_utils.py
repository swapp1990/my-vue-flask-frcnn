import os
import tempfile
import math
import json
import six
from skimage import io, transform

from keras.models import load_model
import numpy as np
from keras import backend as K

def find_layer_idx(model, layer_name):
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break
    if layer_idx is None:
        raise ValueError("No layer with name '{}' within the model".format(layer_name))
    return layer_idx

def reverse_enumerate(iterable):
    """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))

def apply_modifications(model, custom_objects=None):
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def add_defaults_to_kwargs(defaults, **kwargs):
    defaults = dict(defaults)
    defaults.update(kwargs)
    return defaults

def random_array(shape, mean=128., std=20.):
     x = np.random.random(shape)
     x = (x - np.mean(x)) / (np.std(x) + K.epsilon())
     x = (x * std) + mean
     return x

def listify(value):
    if not isinstance(value, list):
        value = [value]
    return value

def deprocess_input(input_array, input_range=(0, 255)):
     # normalize tensor: center on 0., ensure std is 0.1
    input_array = input_array.copy()
    input_array -= input_array.mean()
    input_array /= (input_array.std() + K.epsilon())
    input_array *= 0.1

    # clip to [0, 1]
    input_array += 0.5
    input_array = np.clip(input_array, 0, 1)

    # Convert to `input_range`
    return (input_range[1] - input_range[0]) * input_array + input_range[0]

def get_img_shape(img):
    if isinstance(img, np.ndarray):
        shape = img.shape
    else:
        shape = K.int_shape(img)

    if K.image_data_format() == 'channels_last':
        shape = list(shape)
        shape.insert(1, shape[-1])
        shape = tuple(shape[:-1])
    return shape

class _BackendAgnosticImageSlice(object):
    """Utility class to make image slicing uniform across various `image_data_format`.
    """

    def __getitem__(self, item_slice):
        """Assuming a slice for shape `(samples, channels, image_dims...)`
        """
        if K.image_data_format() == 'channels_first':
            return item_slice
        else:
            # Move channel index to last position.
            item_slice = list(item_slice)
            item_slice.append(item_slice.pop(1))
            return tuple(item_slice)
"""Slice utility to make image slicing uniform across various `image_data_format`.
Example:
    conv_layer[utils.slicer[:, filter_idx, :, :]] will work for both `channels_first` and `channels_last` image
    data formats even though, in tensorflow, slice should be conv_layer[utils.slicer[:, :, :, filter_idx]]
"""
slicer = _BackendAgnosticImageSlice()

def stitch_images(images, margin=5, cols=5):
    """Utility function to stitch images together with a `margin`.

    Args:
        images: The array of 2D images to stitch.
        margin: The black border margin size between images (Default value = 5)
        cols: Max number of image cols. New row is created when number of images exceed the column size.
            (Default value = 5)

    Returns:
        A single numpy image array comprising of input images.
    """
    if len(images) == 0:
        return None

    h, w, c = images[0].shape
    n_rows = int(math.ceil(len(images) / cols))
    n_cols = min(len(images), cols)

    out_w = n_cols * w + (n_cols - 1) * margin
    out_h = n_rows * h + (n_rows - 1) * margin
    stitched_images = np.zeros((out_h, out_w, c), dtype=images[0].dtype)

    for row in range(n_rows):
        for col in range(n_cols):
            img_idx = row * cols + col
            if img_idx >= len(images):
                break

            stitched_images[(h + margin) * row: (h + margin) * row + h,
                            (w + margin) * col: (w + margin) * col + w, :] = images[img_idx]

    return stitched_images

def get_num_filters(layer):
    """Determines the number of filters within the given `layer`.

    Args:
        layer: The keras layer to use.

    Returns:
        Total number of filters within `layer`.
        For `keras.layers.Dense` layer, this is the total number of outputs.
    """
    # Handle layers with no channels.
    if K.ndim(layer.output) == 2:
        return K.int_shape(layer.output)[-1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    return K.int_shape(layer.output)[channel_idx]

def load_img(path, grayscale=False, target_size=None):
    img = io.imread(path, grayscale)
    if target_size:
        img = transform.resize(img, target_size, preserve_range=True)
    return img

def normalize(array, min_value=0., max_value=1.):
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + K.epsilon())
    return (max_value - min_value) * normalized + min_value
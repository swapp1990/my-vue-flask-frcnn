import os
import tempfile
import math
import json
import six

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
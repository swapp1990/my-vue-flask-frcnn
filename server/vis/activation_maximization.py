from __future__ import absolute_import

import numpy as np
from keras import backend as K
from util import viz_utils
from vis.optimizer import Optimizer
from vis.losses import ActivationMaximization

def visualize_activation(model, layer_idx, filter_indices=None, seed_input=None, act_max_weight=1, input_range=(0, 255), **optimizer_params):
    input_tensor = model.input
    # Default optimizer kwargs.
    optimizer_params = viz_utils.add_defaults_to_kwargs({
        'seed_input': seed_input,
        'max_iter': 200,
        'verbose': False
    }, **optimizer_params)

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), act_max_weight)
    ]

    opt = Optimizer(input_tensor, losses)
    img = opt.minimize(**optimizer_params)
    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

    return img
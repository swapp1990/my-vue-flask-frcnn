from __future__ import absolute_import

import numpy as np
from keras import backend as K
from util import viz_utils
from vis.optimizer import Optimizer
from vis.losses import ActivationMaximization, LPNorm, TotalVariation

def initOptimizer(model, layer_idx, class_indices=None, l2_norm=True):
    actmaxloss_contrib= 1
    lp_norm_weight = 10
    tv_weight = 10
    input_tensor = model.input
    losses = [
        (ActivationMaximization(model.layers[layer_idx], class_indices), actmaxloss_contrib)
    ]
    if l2_norm:
        losses.append((LPNorm(model.input), lp_norm_weight))
        losses.append((TotalVariation(model.input), tv_weight))
    
    opt = Optimizer(input_tensor, losses)
    print("init optim")
    return opt

def modifyOpt(opt, model, layer_idx, class_indices=None, l2_norm=True):
    actmaxloss_contrib= 1
    lp_norm_weight = 10
    tv_weight = 10
    input_tensor = model.input
    losses = [
        (ActivationMaximization(model.layers[layer_idx], class_indices), actmaxloss_contrib)
    ]
    if l2_norm:
        losses.append((LPNorm(model.input), lp_norm_weight))
        losses.append((TotalVariation(model.input), tv_weight))
    opt.updateLosses(input_tensor, losses)

def visualize_activation_single(opt, input_range=(0, 255)):
    #minimizes the loss using iterations
    img = opt.minimizeSingle()
    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

    return img

#For the given model (imagenet trained) and given layer, maximize the activations and return the image generated. We use loss functions to maximize it using class indices.
def visualize_activation(model, layer_idx, class_indices=None, seed_input=None, input_range=(0, 255), max_iter=200, **optimizer_params):
    actmaxloss_contrib=1
    lp_norm_weight=10
    tv_weight = 10

    input_tensor = model.input
    # Default optimizer kwargs.
    optimizer_params = viz_utils.add_defaults_to_kwargs({
        'seed_input': seed_input,
        'max_iter': max_iter,
        'verbose': False
    }, **optimizer_params)

    #actmaxloss_contrib is the contribution of ActivationMaximization loss to the overall loss. Default is 1
    #class_indices is a list or single index of imagenet class to maximize
    losses = [
        (ActivationMaximization(model.layers[layer_idx], class_indices), actmaxloss_contrib),
        (LPNorm(model.input), lp_norm_weight),

    ]

    #init optimizer with a list of losses
    opt = Optimizer(input_tensor, losses)
    #minimizes the loss using iterations
    img = opt.minimize(**optimizer_params)
    # If range has integer numbers, cast to 'uint8'
    if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        img = np.clip(img, input_range[0], input_range[1]).astype('uint8')

    return img
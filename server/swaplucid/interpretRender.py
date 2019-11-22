from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()
from builtins import range

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import logging
from decorator import decorator

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show, load
import lucid.optvis.render as render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Original Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def score_f(logit, name):
    if name is None:
        return 0
    elif name in model.labels:
        return 

def channel_attr():
    #Init
    img = load("https://storage.googleapis.com/lucid-static/building-blocks/examples/dog_cat.png")
    #plt.imshow(img)
    #plt.show()
    model = models.InceptionV1()
    model.load_graphdef()
    layer = "mixed4d"
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        #placeholder is a place in memory where we will store values during session
        t_input = tf.placeholder_with_default(img, [None, None, 3])
        T = render.import_model(model, t_input, t_input)

        #Compute activations
        acts = T(layer).eval()
        
        #Compute gradient
        logit_all = T("softmax2_pre_activation")[0] #1008 values (for imagenet classes)
        #find logit values for given classes
        logit_1 = logit_all[model.labels.index("Labrador retriever")]
        logit_2 = logit_all[model.labels.index("tiger cat")]
        score = logit_1 - logit_2
        t_grad = tf.gradients([score], [T(layer)])[0]
        grad = t_grad.eval()

        #attribution of y to x is rate at which x changes y times value of x
        #ie rate at which the loss (grad), which is the difference between the 2 classes changes the given layer.
        #We 'eval' it s othe model is already trained, so this should be accurate
        attr = (grad*acts)[0] #attr shape - (14, 14, 528): (14x14) is the kernel size, and 528 is n_channels
        print(attr.shape)

        #reduce down to channels
        ch_attr = attr.sum(0).sum(0)
        #attr shape after first sum - (14, 528), after second sum - (, 528)
        n_show=3
        ns_pos = list(np.argsort(-ch_attr)[:n_show])
        print(ns_pos)
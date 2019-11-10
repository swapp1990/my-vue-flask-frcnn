from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()
from builtins import range

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import logging

from lucid.optvis import objectives, param, transform
from lucid.misc.io import show
from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad
from lucid.misc.gradient_override import gradient_override_map
import global_vars as G

def make_t_image(param_f):
    if param_f is None:
        t_image = param.image(128)
    return t_image

def make_transform_f(transforms):
    if type(transforms) is not list:
        #By default all Lucid transforms to image (Jitter, random_rotate, random_scale etc.) are applied
        transforms = transform.standard_transforms
    transforms_f = transform.compose(transforms)
    #Returns a tf func which is run on sess.run() on the input image
    return transforms_f

def make_optimizer(optimizer, args):
    if optimizer is None:
        return tf.train.AdamOptimizer(0.05)

def import_model(model, t_image, t_image_raw=None, scope="import"):
    #modelzoo vision_base.py 
    #Uses tf.import_graph_def which loads the graph def for the given 'model' (InceptionV1)
    model.import_graph(t_image, scope=scope, forget_xy_shape=True)
    #Layer is passed during tf.sess
    #T("input") - the raw image
    #
    def model_tensor_f(layer):
        if layer == "input": return t_image_raw
        if layer == "labels": return model.labels
        if ":" in layer:
            return t_image.graph.get_tensor_by_name("%s/%s" % (scope,layer))
        else:
            return t_image.graph.get_tensor_by_name("%s/%s:0" % (scope,layer))

    return model_tensor_f

def make_vis_T(model, objective_f):
    t_image = make_t_image(None)
    #print(t_image) #Tensor (1,128,18,3) - The initial image def (?)
    objective_f = objectives.as_objective(objective_f)
    #Converts string into lucid.Objective class (?)
    transforms_f = make_transform_f(None)
    optimizer = make_optimizer(None, [])

    global_step = tf.train.get_or_create_global_step()
    init_global_step = tf.variables_initializer([global_step])
    init_global_step.run()

    #returns the tensor for the given layer while tf.sess
    T = import_model(model, transforms_f(t_image), t_image)
    #calculate the loss for the tensor using objective function
    loss = objective_f(T)

    vis_op = optimizer.minimize(-loss, global_step=global_step)

    #used for pulling out local vars while each step is teaking place 
    local_vars = locals()

    #Called while tf.sess with input 'name' is the layer name in string
    def final_T_f(name):
        if name in local_vars:
            #diectly returns the vars values instead of performing any tensor operation
            return local_vars[name]
        return T(name)
    return final_T_f

def swapShow(thing):
    if isinstance(thing, np.ndarray):
        rank = len(thing.shape)
        if rank in (2, 3):
            plt.imshow(thing)
            plt.show()

def initSess(model, objective_f):
    G.graph = tf.Graph()
    with G.graph.as_default(), tf.Session(graph=G.graph) as sess:
        T = make_vis_T(model, objective_f)
        #returns a function which whe called returns a Tensor (image) after loss has been applied
        G.loss, G.vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()
        print("Init sess")

def runStepSess():
    with tf.Session(graph=G.graph) as sess:
        G.loss_, _ = sess.run([G.loss, G.vis_op])
        print(G.loss)

def render_vis_yield(model, objective_f, max_steps=10):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        T = make_vis_T(model, objective_f)
        #returns a function which whe called returns a Tensor (image) after loss has been applied
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()

        i = 1
        img = None
        while i < max_steps:
            loss_, _ = sess.run([loss, vis_op])
            vis = t_image.eval()
            img = np.hstack(vis)
            yield img
            i += 1

def render_vis(model, objective_f):
    param_f=None
    optimizer=None
    transforms=None 
    thresholds=(400,)
    
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        T = make_vis_T(model, objective_f)
        #returns a function which whe called returns a Tensor (image) after loss has been applied
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()

        images = []
        img = None
        try:
            for i in range(max(thresholds)+1):
                loss_, _ = sess.run([loss, vis_op])
                if i in [50,100,250,400,500]:
                    print(i)
                    vis = t_image.eval()
                    emit("test")
                    #swapShow(np.hstack(vis))
                if i in thresholds:
                    vis = t_image.eval()
                    images.append(vis)
                    img = np.hstack(vis)
        except KeyboardInterrupt:
            print("Terminated")
        return img
from __future__ import absolute_import, division, print_function
from future.standard_library import install_aliases
install_aliases()
from builtins import range

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import logging
from decorator import decorator

from lucid.optvis import objectives, param, transform
from lucid.optvis.param.spatial import fft_image
from lucid.optvis.param.color import to_valid_rgb
from lucid.misc.io import show
from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad
from lucid.misc.gradient_override import gradient_override_map
import global_vars as G

def myimagef(w, h=None, batch=None):
    h = h or w
    batch = batch or 1
    shape = [batch, h, w, 3]
    init_val = np.random.normal(size=shape, scale=0.01).astype(np.float32)
    pixel_image = tf.Variable(init_val)
    param_f = pixel_image
    #t = param_f(shape, sd=None)
    t = param_f
    output = to_valid_rgb(t[..., :3], decorrelate=True, sigmoid=True)
    return output

def make_t_image(param_f):
    if param_f is None:
        t_image = myimagef(128)
    elif callable(param_f):
        t_image = param_f()
    elif isinstance(param_f, tf.Tensor):
        t_image = param_f
    else:
        print('Error! Param_f')
        t_image = None
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

def my_objective_func(layer, n_channel, batch=None):
    def inner(T):
        t = T(layer)
        if isinstance(batch, int):
            t = t[batch:batch+1]
        return tf.reduce_mean(t[...,n_channel])
    return inner

#Returns a objective func which subtracts two obj functions
def my_diff_objs_f(obj1, obj2):
    def inner(T):
        t1 = obj1(T)
        t2 = obj2(T)
        tf = t1 - t2
        return tf
    return inner

def make_vis_T(model, objective_f, param_f=None):
    #make original image
    t_image = make_t_image(param_f)
    #print(t_image) #Tensor (1,128,18,3) - The initial image def (?)

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

    vis_op = optimizer.minimize(loss, global_step=global_step)

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

###################################### Client Socket Yield ###########################################
def createObjective_f(layer_name, filter_idx, batch=None):
    #"mixed4a_pre_relu"
    if batch is 2:
        obj1 = my_objective_func(layer_name, filter_idx, batch=1)
        obj2 = my_objective_func(layer_name, filter_idx, batch=0)
        obj_f = my_diff_objs_f(obj1, obj2)
        return obj_f
    else:
        return my_objective_func(layer_name, filter_idx, batch=None)

def render_vis_yield(model, layer_name, filter_idx, show_negative=False, max_steps=10):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        batch = 2 if show_negative else None
        objective_f = createObjective_f(layer_name, filter_idx, batch)
        #We need to show 2 images in the final viz, param_f modifies the initial display image
        param_f = None
        if show_negative:
            param_f = myimagef(128, batch=2)
        T = make_vis_T(model, objective_f, param_f)
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

###################################### Client Socket Yield ###########################################

#objective to minimize the loss on
#param
def render_vis(model, objective_f, param_f=None):
    param_f=None
    optimizer=None
    transforms=None 
    thresholds=(400,)
    
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        #Test
        #param_f = 
        T = make_vis_T(model, objective_f, param_f)
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
                #     vis = t_image.eval()
                #     emit("test")
                    #swapShow(np.hstack(vis))
                if i in thresholds:
                    vis = t_image.eval()
                    images.append(vis)
                    img = np.hstack(vis)
        except KeyboardInterrupt:
            print("Terminated")
        return img

def render_vis_test(model):
    """Visualize a single channel"""
    # obj = objectives.channel("mixed4a_pre_relu", 492, batch=1) - objectives.channel("mixed4a_pre_relu", 492, batch=0)
    obj = objectives.channel("mixed4a_pre_relu", 492)
    #img = render_vis(model, "mixed4a_pre_relu:476")
    img = render_vis(model, obj)
    plt.imshow(img)
    plt.show()
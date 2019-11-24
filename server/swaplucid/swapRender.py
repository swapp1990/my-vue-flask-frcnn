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

from .myobjectives import direction_obj, channel_obj_f, L1, total_variation, blur_input_each_step, sum_objs_f, mul_obj_f
import global_vars as G

def myimagef(w, h=None, batch=None):
    h = h or w
    batch = batch or 1
    shape = [batch, h, w, 3]
    #init_val = np.random.normal(size=shape, scale=0.01).astype(np.float32)
    #pixel_image = tf.Variable(init_val)
    param_f = fft_image
    t = param_f(shape, sd=None)
    #t = param_f
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

###################################### Diversity Obj Func Yield #####################################
#Must always return a Tensor 'T(layer)' (processed with the objective func)
def diversity_obj_f(layer):
    """Encourage diversity between each batch element.
    Calculates the correlation matrix of activations at layer for each given image (from batch), when that image is passed through this layer. Then it penalizes cosine similarity between them.
    The func returns ?
    """
    def inner(T):
        layer_t = T(layer)
        batch_n, _, _, channels = layer_t.get_shape().as_list()
        #(4,25 (5*5), n_channels)
        flattened = tf.reshape(layer_t, [batch_n, -1, channels])
        grams = tf.matmul(flattened, flattened, transpose_a=True)
        grams = tf.nn.l2_normalize(grams, axis=[1,2], epsilon=1e-10)
        grams_sum = sum([ sum([ tf.reduce_sum(grams[i]*grams[j])
                      for j in range(batch_n) if j != i])
                for i in range(batch_n)]) / batch_n

        # for i in range(batch_n)
        #     for j in range(batch_n):
        #         if j != i:
        #            grams_sum = sum(tf.reduce_sum(grams[i]*grams[j]), grams_sum)

        return grams_sum
    return inner

def diversity_render_yield(model, filter_idx, layer, n_batch=2):
    #Init
    param_f = lambda: param.image(128, batch=n_batch)
    #Only diversity obj
    #objective_f = diversity_obj_f("mixed5a")
    #diff between channel obj and diversity obj
    print("obj1=" + "mixed4e_pre_relu:" +str(filter_idx))
    obj1 = channel_obj_f("mixed4e_pre_relu", filter_idx)
    obj2 = diversity_obj_f("mixed4e")
    obj_f = my_diff_objs_f(obj1, obj2, subWeight=100)

    with tf.Graph().as_default() as graph, tf.Session() as sess:
        T = make_vis_T(model, obj_f, param_f)
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()
        i = 1
        img = None
        while i < 500:
            loss_, _ = sess.run([loss, vis_op])
            vis = t_image.eval()
            img = np.hstack(vis)
            yield img
            i += 1
####################################################################################################

###################################### Neurons Add Obj Func Yield #####################################
#returns function that sums up 2 neuron objective functions
def sum_2_neuron_obj_f(obj1, obj2, addWeight=1.):
    def inner(T):
        t1 = obj1(T)
        t2 = obj2(T)
        tf = t1 + addWeight*t2
        return tf
    return inner

def create_neuron_obj_f(neurons):
    n0 = neurons[0]
    n1 = neurons[1]
    obj1 = channel_obj_f(n0['layer'], n0['filterIndex'])
    obj2 = channel_obj_f(n1['layer'], n1['filterIndex'])
    obj_f = sum_2_neuron_obj_f(obj1, obj2)
    return obj_f

def render_vis_neurons_yield(model, neurons, max_steps=10):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        batch = None
        objective_f = create_neuron_obj_f(neurons)
        #We need to show 2 images in the final viz, param_f modifies the initial display image
        param_f = myimagef(128, batch=1)
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

#Returns a objective func which subtracts two obj functions
def my_diff_objs_f(obj1, obj2, subWeight=1.):
    def inner(T):
        t1 = obj1(T)
        t2 = obj2(T)
        tf = t1 - subWeight*t2
        return tf
    return inner

def test_obj_f(model):
    #Init
    param_f = lambda: param.image(128, batch=1)
    objective_f = diversity_obj_f("mixed5a")

    with tf.Graph().as_default() as graph, tf.Session().as_default() as sess:
        T = make_vis_T(model, objective_f, param_f)
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()
        for i in range(500):
            loss_, _ = sess.run([loss, vis_op])
            evalLoss = loss.eval()
            if i in [50,100,250,400,500]:
                vis = t_image.eval()
                img = np.hstack(vis)
                plt.imshow(img)
                plt.show()

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

    #Whether to use the gradient override scheme
    with gradient_override_map({'Relu': redirected_relu_grad,
                                'Relu6': redirected_relu6_grad}):
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

###################################### Client Socket Yield ###########################################
def createObjective_f(layer_name, filter_idx, batch=None):
    #"mixed4a_pre_relu"
    if batch is 2:
        obj1 = channel_obj_f(layer_name, filter_idx, batch=1)
        obj2 = channel_obj_f(layer_name, filter_idx, batch=0)
        obj_f = my_diff_objs_f(obj1, obj2)
        return obj_f
    else:
        return channel_obj_f(layer_name, filter_idx, batch=None)

#Only for batch 1 for now
def createObjWithRegularization(layer_name, filter_idx, params=None):
    obj1 = channel_obj_f(layer_name, filter_idx, batch=None)
    #penalize     #obj2 = L1(constant=.5)
    L1_const = .5
    L1_weight = -0.05
    TV_weight = -0.25
    Blur_weight = 0.
    if params is not None:
        L1_const = float(params["L1_const"])
        L1_weight = float(params["L1_weight"])
        TV_weight = float(params["TV_weight"])
        Blur_weight = float(params["Blur_weight"])

    print(L1_const, L1_weight, TV_weight, Blur_weight)
    obj2 = mul_obj_f(L1(constant=L1_const), L1_weight)
    obj3 = mul_obj_f(total_variation(), TV_weight)
    #obj4 = mul_obj_f(blur_input_each_step(), Blur_weight)
    objs = [obj1, obj2, obj3]
    obj_f = sum_objs_f(objs)
    return obj_f

def render_vis_direction_yield(model, layer_name, max_steps=10):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        random_dir = np.random.randn(528)
        objective_f = direction_obj(layer_name, random_dir)
        param_f = myimagef(128, batch=1)
        T = make_vis_T(model, objective_f, param_f)
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

def render_vis_yield(model, layer_name, filter_idx, show_negative=False, use_regularizer=False, regularizer_params=None, max_steps=10):
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        batch = 2 if show_negative else None
        if use_regularizer:
            objective_f = createObjWithRegularization(layer_name, filter_idx, regularizer_params)
        else:
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
        try:
            while i < max_steps:
                loss_, _ = sess.run([loss, vis_op])
                vis = t_image.eval()
                img = np.hstack(vis)
                yield img
                i += 1
        except:
            print("Exception in Generator")
            return
        finally:
            print("Generator Ended")
            return
            
###################################### Client Socket Yield ###########################################

#objective to minimize the loss on
#param
def render_vis(model, objective_f, param_f=None):
    #param_f=None
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
    # obj = objectives.channel("mixed4a_pre_relu", 492)
    param_f = lambda: param.image(128, batch=4)
    obj = objectives.channel("mixed5a_pre_relu", 9) - 1e2*objectives.diversity("mixed5a")
    #img = render_vis(model, "mixed4a_pre_relu:476")
    img = render_vis(model, obj, param_f)
    plt.imshow(img)
    plt.show()
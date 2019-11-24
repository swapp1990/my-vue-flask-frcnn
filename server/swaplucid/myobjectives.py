import tensorflow as tf
import numpy as np

#This objective gives loss for the given generated image per channel for the layer
def channel_obj_f(layer, n_channel, batch=None):
    def inner(T):
        t = T(layer)
        if isinstance(batch, int):
            t = t[batch:batch+1]
        return tf.reduce_mean(t[...,n_channel])
    return inner

#Objective to find and visualize a direction for the given layer
def direction_obj(layer, vec):
    vec = vec[None, None, None]
    vec = vec.astype("float32")

    def inner(T):
        return dot_cossim(T(layer), vec)
    return inner

######################### Regularization Objectives #############################
def L1(layer="input", constant=0, batch=None):
     """L1 norm of layer. Generally used as penalty."""
     return lambda T: tf.reduce_sum(tf.abs(T(layer) - constant))

def total_variation(layer="input"):
     """Total variation of image (or activations at some layer).
     This operation is most often used as a penalty to reduce noise."""
     return lambda T: tf.image.total_variation(T(layer))
    
def blur_input_each_step():
    def inner(T):
        t_input = T("input")
        t_input_blurred = tf.stop_gradient(_tf_blur(t_input))
        return 0.5*tf.reduce_sum((t_input - t_input_blurred)**2)
    return inner

########## operations on multiple objectives
#Returns functions that sums up result of given objectives (list)
def sum_objs_f(objList):
    def inner(T):
        tf = objList[0](T)
        for i in range(len(objList)):
            if i > 0:
                t = objList[i](T)
                tf = tf + t
        return tf
    return inner

def mul_obj_f(obj, val=1):
    def inner(T):
        tf = val * obj(T)
        return tf
    return inner
###################################### Utils ############################################################
def _dot(x, y):
  return tf.reduce_sum(x * y, -1)

def dot_cossim(x,y,cossim_pow=0):
    eps = 1e-4
    xy_dot = _dot(x,y)
    if cossim_pow == 0: return tf.reduce_mean(xy_dot)

def _tf_blur(x, w=3):
  depth = x.shape[-1]
  k = np.zeros([w, w, depth, depth])
  for ch in range(depth):
    k_ch = k[:, :, ch, ch]
    k_ch[ :,    :  ] = 0.5
    k_ch[1:-1, 1:-1] = 1.0

  conv_k = lambda t: tf.nn.conv2d(t, k, [1, 1, 1, 1], "SAME")
  return conv_k(x) / conv_k(tf.ones_like(x))


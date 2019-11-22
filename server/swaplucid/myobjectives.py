import tensorflow as tf

#Objective to find and visualize a direction for the given layer
def direction_obj(layer, vec):
    vec = vec[None, None, None]
    vec = vec.astype("float32")

    def inner(T):
        return dot_cossim(T(layer), vec)
    return inner
###################################### Utils ############################################################
def _dot(x, y):
  return tf.reduce_sum(x * y, -1)

def dot_cossim(x,y,cossim_pow=0):
    eps = 1e-4
    xy_dot = _dot(x,y)
    if cossim_pow == 0: return tf.reduce_mean(xy_dot)



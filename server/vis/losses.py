from __future__ import absolute_import
from keras import backend as K
from util import viz_utils
import numpy as np

def normalize(input_tensor, output_tensor):
    """Normalizes the `output_tensor` with respect to `input_tensor` dimensions.
    This makes regularizer weight factor more or less uniform across various input image dimensions.
    """
    image_dims = viz_utils.get_img_shape(input_tensor)[1:]
    return output_tensor / np.prod(image_dims)

class Loss(object):
    def __init__(self):
        self.name = "Unnamed Loss"
    
    def __str__(self):
        return self.name

    def build_loss(self):
        raise NotImplementedError()

class ActivationMaximization(Loss):
    def __init__(self, layer, filter_indices):
        super(ActivationMaximization, self).__init__()
        self.name = "AM"
        self.layer = layer
        self.filter_indices = viz_utils.listify(filter_indices)
    
    def build_loss(self):
        layer_output = self.layer.output
        #print(layer_output)
        is_dense = K.ndim(layer_output) == 2
        #print("is_dense ", is_dense)
        loss = 0.
        for idx in self.filter_indices:
            loss += -K.mean(layer_output[:, idx])
       # print("loss ", loss)
        return loss

#Regularizers
class LPNorm(Loss):
    def __init__(self, img_input, p=6.):
        """ This regularizer encourages the intensity of pixels to stay bounded.
            i.e., prevents pixels from taking on very large values.
        """
        super(LPNorm, self).__init__()
        if p < 1:
            raise ValueError('p value should range between [1, inf)')
        self.name = "LP"
        # self.name = "L-{} Norm Loss".format(p)
        self.p = p
        self.img = img_input

    def build_loss(self):
        if np.isinf(self.p):
            value = K.max(self.img)
        else:
            value = K.pow(K.sum(K.pow(K.abs(self.img), self.p)), 1. / self.p)
        
        return normalize(self.img, value)

class TotalVariation(Loss):
    def __init__(self, img_input, beta=2.):
        """Total variation regularizer encourages blobbier and coherent image structures, akin to natural images.
        """
        super(TotalVariation, self).__init__()
        self.name = "TV"
        #self.name = "TV({}) Loss".format(beta)
        self.img = img_input
        self.beta = beta

    def build_loss(self):
        image_dims = K.ndim(self.img) - 2
         # Constructing slice [1:] + [:-1] * (image_dims - 1) and [:-1] * (image_dims)
        start_slice = [slice(1, None, None)] + [slice(None, -1, None) for _ in range(image_dims - 1)]
        end_slice = [slice(None, -1, None) for _ in range(image_dims)]
        samples_channels_slice = [slice(None, None, None), slice(None, None, None)]

        # Compute pixel diffs by rolling slices to the right per image dim.
        tv = None
        for i in range(image_dims):
            ss = tuple(samples_channels_slice + start_slice)
            es = tuple(samples_channels_slice + end_slice)
            diff_square = K.square(self.img[viz_utils.slicer[ss]] - self.img[viz_utils.slicer[es]])
            tv = diff_square if tv is None else tv + diff_square

            # Roll over to next image dim
            start_slice = np.roll(start_slice, 1).tolist()
            end_slice = np.roll(end_slice, 1).tolist()
        
        tv = K.sum(K.pow(tv, self.beta / 2.))
        return normalize(self.img, tv)
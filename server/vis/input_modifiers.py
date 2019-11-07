from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import shift
from keras import backend as K

from util import viz_utils
from matplotlib import pyplot as plt

class InputModifier(object):
    def pre(self, inp):
        return inp
    
    def post(self, inp):
        return inp

class Jitter(InputModifier):
    def __init__(self, jitter=0.05):
        """ Modifies the input image to introduce random 'jitter' which produces crisper activation maximization """
        super(Jitter, self).__init__()
        self.jitter = np.array(viz_utils.listify(jitter))
        if np.any(jitter < 0.):
            raise ValueError('Jitter value should be positive')
        self._processed = False

    def _process_jitter_values(self, image_dims):
        if len(self.jitter) == 1:
            self.jitter = np.repeat(self.jitter, len(image_dims))
        
        # Convert percentage to absolute values.
        for i, jitter_val in enumerate(self.jitter):
            if jitter_val < 1.:
                self.jitter[i] = image_dims[i] * jitter_val
        
        # Round to int.
        self.jitter = np.int32(self.jitter)
        self._processed = True

    def pre(self, img):
        if not self._processed:
            image_dims = viz_utils.get_img_shape(img)[2:] #(244,244)
            self._process_jitter_values(image_dims)

        dim_offsets = [np.random.randint(-value, value+1) for value in self.jitter]

        if K.image_data_format() == 'channels_first':
            shift_vector = np.array([0, 0] + dim_offsets)
        else:
            shift_vector = np.array([0] + dim_offsets + [0])

        img = shift(img, shift_vector, mode='wrap', order=0)

        # input_range=(0, 255)
        # img = viz_utils.deprocess_input(img[0], input_range)
        # if isinstance(input_range[0], int) and isinstance(input_range[1], int):
        #     img = np.clip(img, input_range[0], input_range[1]).astype('uint8')
        # plt.imshow(img)
        # plt.show()
        return img


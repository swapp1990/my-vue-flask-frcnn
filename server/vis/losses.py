from __future__ import absolute_import
from keras import backend as K
from util import viz_utils

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
        self.name = "ActivationMax Loss"
        self.layer = layer
        print("ActivationMaximization ", filter_indices)
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

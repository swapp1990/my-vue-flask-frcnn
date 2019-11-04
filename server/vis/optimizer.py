from __future__ import absolute_import

import numpy as np
from keras import backend as K
from util import viz_utils

class Optimizer(object):

    def __init__(self, input_tensor, losses, input_range=(0, 255)):
        """Creates an optimizer that minimizes weighted loss function.
        Args:
            losses: List of ([Loss](vis.losses.md#Loss), weight) tuples.
        """
        self.input_tensor = input_tensor
        self.input_range = input_range
        self.seed_input = None
        self.cache=None
        self.best_loss = float('inf')
        self.grads = None
        self.wrt_tensor = self.input_tensor
        if self.input_tensor is self.wrt_tensor:
            self.wrt_tensor_is_input_tensor = True
            self.wrt_tensor = K.identity(self.wrt_tensor)
        self.loss_functions = []
        self.loss_names = []
        overall_loss = None
        for loss, contrib in losses:
            if contrib != 0:
                 loss_fn = contrib * loss.build_loss()
                 self.loss_functions.append(loss_fn)
                 overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
                 self.loss_names.append(loss.name)
        
        if self.wrt_tensor_is_input_tensor:
            grads = K.gradients(overall_loss, self.input_tensor)[0]
        # K.function just glues the inputs to the outputs returning a single operation that when given the inputs it will follow the computation graph from the inputs to the defined outputs. It's symbolic.
        self.compute_fn = K.function([self.input_tensor, K.learning_phase()], self.loss_functions + [overall_loss, grads, self.wrt_tensor])

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_input(self, seed_input):
        """Creates a random seed_input if None """
        if seed_input is None:
            desired_shape = (1, ) + K.int_shape(self.input_tensor)[1:]
            return viz_utils.random_array(desired_shape, mean=np.mean(self.input_range),
                                      std=0.05 * (self.input_range[1] - self.input_range[0]))
        else:
            return seed_input
    
    def minimizeSingle(self):
        self.seed_input = self._get_seed_input(self.seed_input)
        computed_values = self.compute_fn([self.seed_input, 0])
        #losses = computed_values[:len(self.loss_names)]
        overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]
        print("overall_loss ", overall_loss)
        if self.wrt_tensor_is_input_tensor:
            step, self.cache = self._rmsprop(grads, self.cache)
            self.seed_input += step

        if overall_loss < self.best_loss:
            self.best_loss = overall_loss.copy()
        best_input = self.seed_input.copy()
        return viz_utils.deprocess_input(best_input[0], self.input_range)

    def minimize(self, seed_input=None, max_iter=10, verbose=True):
        seed_input = self._get_seed_input(seed_input)
        print("seed_input", seed_input.shape)
        cache=None
        best_loss = float('inf')
        best_input = None
        grads = None
        wrt_value = None

        for i in range(max_iter):
            # 0 learning phase for 'test'
            
            computed_values = self.compute_fn([seed_input, 0])
            #print(computed_values[0]) acm loss

            losses = computed_values[:len(self.loss_names)]
            named_losses = list(zip(self.loss_names, losses))
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]
            print("overall_loss ", overall_loss)
            if self.wrt_tensor_is_input_tensor:
                step, cache = self._rmsprop(grads, cache)
                seed_input += step
            
            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_input = seed_input.copy()
        
        return viz_utils.deprocess_input(best_input[0], self.input_range)
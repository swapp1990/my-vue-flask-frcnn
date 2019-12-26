import os
import logging
import io
import base64
from time import time
from collections import namedtuple

import tensorflow as tf
from tensorflow import keras
print("tf version {}".format(tf.__version__))
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

import numpy as np 
import PIL.Image
from matplotlib import pyplot as plt

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
import mpld3

######################################### Instantiate ######################################
# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

######################################### Constants ######################################


######################################### Inception ########################################

class Inception():
    def __init__(self):
        #Number of iterations of backprop to perform on the input noise
        self.renderIter_n = 128
        #step size for backprop operation on gradient of the image
        self.step_size = 0.0015 
        self.saturation = 0.6
        self.shift = 32
        self.tilesize = 4
        self.batchToProcess = 8
        self.layer_name = 'mixed4b'
        self.possibleLayers = []
        self.graph = self.loadInception()
        self.feature_n = self.testRunModel()

    def loadInception(self):
        data = open('InceptionV1.pb', 'rb').read()
        graph_def = tf.compat.v1.GraphDef.FromString(data)
        #remove layers with 'avgpool0'
        i = [n.name for n in graph_def.node].index('avgpool0')
        del graph_def.node[i:]
        return graph_def
    
    def modifyModel(self):
        self.layer_name = 'mixed4c'
        print("modify " + self.layer_name)

    def _populate_inception_bottleneck(self, graph, scope):
        """Add inception bottlenecks and their pre-relu versions to the graph."""
        for op in graph.get_operations():
            #print(op.name, op.type)
            if op.name.startswith(scope+'/') and 'Concat' in op.type:
                # print(op.name, op.type)
                name = op.name.split('/')[1]
                pre_relus = []
                for tower in op.inputs[1:]:
                    if tower.op.type == 'Relu':
                        tower = tower.op.inputs[0]
                    pre_relus.append(tower)
                concat_name = scope + '/' + name + '_pre_relu'
                #print(concat_name)
                self.possibleLayers.append(name)
                #This is needed for retreiveing the new tensor created of concat_name
                _ = tf.concat(pre_relus, -1, name=concat_name)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None,None,None,3], dtype=tf.float32)])
    @tf.function()
    def run_model(self, inputs, layer_name):
        image_value_range = (-117.0, 255-117.0)
        lo, hi = image_value_range
        prep_inputs = inputs*(hi-lo)+lo
        tf.import_graph_def(self.graph, {'input': prep_inputs})
        g = tf.compat.v1.get_default_graph()
        self._populate_inception_bottleneck(g, 'import')
        t = g.get_tensor_by_name('import/%s_pre_relu:0'%layer_name)
        return t
    
    def testRunModel(self):
        t = self.run_model(tf.zeros([1,128,128,3]), layer_name=self.layer_name)
        feature_n = t.shape[-1]
        print(self.layer_name + "-" + str(feature_n))
        return feature_n
    
    def getPyramidLevels(self, pyramidLevels=4):
        imgs = []
        startingRes = 16
        std = 0.01
        batch_size = self.batchToProcess
        for i in range(pyramidLevels):
            imgs.append(tf.random.normal([batch_size, startingRes, startingRes, 3])*std)
            startingRes = startingRes*2
        return imgs
    
    def merge_pyramid(self, pyr, saturation=0.5):
        img = pyr[0] + [0.5,0.0,0.0]
        for hi in pyr[1:]:
            hw = tf.shape(hi)[1:3]
            img = tf.image.resize(img, hw) + hi
        return tf.image.yuv_to_rgb(img*[1.0, saturation, saturation])

    def render(self, fm_i = 0, pyramidLevels=4):
        self.feature_n = self.testRunModel()
        pyr = self.getPyramidLevels(pyramidLevels)
        i = 0
        #List of fm_idxs
        fm_idxs = [fm_i+k for k in range(self.batchToProcess)]
        fm_is = tf.constant(fm_idxs)
        #print(tf.one_hot(x%5,5))
        while i < self.renderIter_n:
            i += 1
            with tf.GradientTape(persistent=True) as g:
                g.watch(pyr)
                img = self.merge_pyramid(pyr, self.saturation)
                shift = tf.random.uniform([2], 0, self.shift, dtype=tf.int32)
                simg = tf.roll(img, shift, [1, 2])
                t = self.run_model(simg, self.layer_name) #(n, 4, 4, 512)
                feature_n = t.shape[-1]
                t = tf.reduce_mean(t, [1,2]) #(n, 512)
                score = tf.reduce_sum(t*tf.one_hot(fm_is%self.feature_n, self.feature_n))
            grad_img = g.gradient(score, img)
            grad_img /= tf.reduce_mean(tf.abs(grad_img), [1, 2, 3], keepdims=True)+1e-8
            grad_pyr = g.gradient(img, pyr, grad_img)
            pyr = [p+g*self.step_size for p, g in zip(pyr, grad_pyr)]
        return self.merge_pyramid(pyr, self.saturation)

    #Renders a single feature map and returns matplot2d3 fig for html rendering
    def renderFeature(self, input_idx=0, inp_layer_name='mixed4a', inp_itern=128, pyramidLevels=4, inp_saturation=0.6, inp_shift = 32):
        self.layer_name = inp_layer_name
        self.saturation = inp_saturation
        self.shift = inp_shift
        self.renderIter_n = inp_itern
        #print(self.renderIter_n)
        res = self.render(fm_i = input_idx, pyramidLevels=pyramidLevels)
        title = self.layer_name + ":" + str(input_idx)
        my_dpi = 96
        img_size = (128*self.tilesize, 128)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        #Add borders between fms
        res = tf.pad(res, [(0, 0), (1, 1), (1, 1), (0, 0)]).numpy()
        img_tile = self.tile2d(res, self.tilesize)
        ax.imshow(img_tile, cmap='plasma')
        mp_fig = mpld3.fig_to_dict(fig)
        # plt.close()
        return mp_fig
        #plt.show()

    def tile2d(self, a, w=None):
        if w is None:
            w = int(np.ceil(np.sqrt(len(a))))
        th, tw = a.shape[1:3]
        pad = (w-len(a))%w
        a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
        h = len(a)//w
        a = a.reshape([h, w]+list(a.shape[1:]))
        a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
        return a
    
    def test_plotFeature(self):
        res = self.render()

        #simple test
        # plt.imshow(res[0])
        # plt.show()

        my_dpi = 96
        img_size = (128*4, 128)
        fig = plt.figure(figsize=(img_size[0]/my_dpi, img_size[1]/my_dpi), dpi=my_dpi)
        ax = fig.add_subplot(111)
        # ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        #Add borders between fms
        res = tf.pad(res, [(0, 0), (1, 1), (1, 1), (0, 0)]).numpy()
        img_tile = self.tile2d(res, 4)
        ax.imshow(img_tile, cmap='plasma')
        plt.show()

inceptionModel = Inception()
#inceptionModel.test_plotFeature()
######################################### Socket ###########################################
class DictToObject(object):
    def __init__(self, dictionary):
        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)

@socketio.on('init')
def init(content):
    #inceptionModel = Inception()
    msg = {"log": "Created Inception Model"}
    emit("General", msg)

@socketio.on('modifyModel')
def modifyModel():
    inceptionModel.modifyModel()

@socketio.on('getFeatureMap')
def getFeatureMap(msg):
    print(msg)
    config = DictToObject(msg)
    featureMapIdx = 0
    if config.featureMapIdx:
        featureMapIdx = int(config.featureMapIdx)
    pyramidLevels = 4
    saturation = 0.6
    shift = 32
    itern = 128
    if config.genDetails:
        itern = int(config.genDetails.itern)
        pyramidLevels = int(config.genDetails.pyramidLevels)
        saturation = float(config.genDetails.saturation)
        shift = int(config.genDetails.shift)
    fig = inceptionModel.renderFeature(featureMapIdx, inp_layer_name=config.layer_name, inp_itern=itern, pyramidLevels=pyramidLevels,inp_saturation=saturation, inp_shift=shift)
    msg = {"action": "showFig", "id": 0, "fig": fig}
    socketio.emit('General', msg)

######################################################################################
if __name__ == "__main__":
    print("running socketio")
    socketio.run(app)
    #inceptionModel = Inception()
    #inceptionModel.renderFeature()
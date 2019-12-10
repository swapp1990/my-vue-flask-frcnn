import logging
import threading
import time
import mpld3
import json
from mpld3 import plugins
import concurrent.futures
import queue
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

import lucid.modelzoo.vision_models as models

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

DATASET_DIR = "C:\\Users\Swapinl\Documents\Datasets\\flowers"
TRAIN_DIR = DATASET_DIR +  "\\train"
VALID_DIR = DATASET_DIR + "\\validation"
IMG_SIZE = (299, 299, 3)
BATCH_SIZE = 32

class Inception():
    def __init__(self, filename):
        self.model = self.loadModel(filename)
        self.currImg = 'daisy'
        self.activations = {}
        self.layers = []
        self.getModelActivations()
        self.currIdxs = []

    def loadModel(self, filename):
        model = load_model(filename)
        print(model.summary())
        return model
    
    def getModelActivations(self):
        self.layers = []
        self.activations = {}
        layer_outputs = [l.output for l in self.model.layers[:50]][1:]
        for idx, layer in enumerate(self.model.layers[:50]):
            self.layers.append({"i": idx, "name": layer.name})

        test_img = 'samples\\'+self.currImg+'.jpg'
        img = image.load_img(test_img, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
        #plotImg(img)
        img_t = image.img_to_array(img)
        img_t = np.expand_dims(img_t, axis=0)
        img_t /= 255.

        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
        self.activations = activation_model.predict(img_t)

def getFigForLayer(layer_idx):
    print(inceptionModel.layers[layer_idx])
    layer_name = inceptionModel.layers[layer_idx]['name']
    layer_fm = inceptionModel.activations[layer_idx]
    fig = showAllChannelsInFeatureMap(layer_name, layer_fm)
    return fig

def getFigsArrayByFilter(filter):
    figs = []
    if filter == "activations":
        filtered_layers = [l for l in inceptionModel.layers if 'activation' in l['name']]
        filtered_layers = filtered_layers[:-1]
        #print(filtered_layers)
        for l in filtered_layers:
            figs.append(getFigForLayer(l["i"]))
        #print(len(figs))
    return figs

def getFigsArrayByIdxs():
    idxs = inceptionModel.currIdxs
    print("idxs ", idxs)
    filtered_layers = [l for l in inceptionModel.layers if l['i'] in idxs]
    figs = []
    for l in filtered_layers:
        figs.append(getFigForLayer(l["i"]))
    print("len figs ", len(figs))
    return figs

def plotImg(img):
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def showAllChannelsInFeatureMap(name, fm):
    images_per_row = 16
    n_features = fm.shape[-1] #32
    #size = fm.shape[1] #299
    size = 128
    n_cols = n_features // images_per_row #2
    display_grid = np.zeros((size * n_cols, size* images_per_row))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_img = fm[0,:,:,col*images_per_row+row]
            channel_img -= channel_img.mean()
            channel_img /= channel_img.std()
            channel_img *= 64
            channel_img += 128
            channel_img = np.clip(channel_img, 0, 255).astype('uint8')
            channel_img = cv2.resize(channel_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_img
    
    scale = 1./size
    #print("size " + str(size) + " scale" + str(scale))
    fig = plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    ax = fig.add_subplot(111)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(display_grid, cmap='plasma')
    #plt.show()
    # mp_fig = mpld3.fig_to_html(fig)
    mp_fig = mpld3.fig_to_dict(fig)
    return mp_fig

def initTest():
    model = load_model('inceptionv3_flowers.h5')
    print(model.summary())
    layer_outputs = [l.output for l in model.layers[:50]][1:]
    test_img = 'daisy.jpg'
    img = image.load_img(test_img, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
    plotImg(img)
    img_t = image.img_to_array(img)
    img_t = np.expand_dims(img_t, axis=0)
    img_t /= 255.

    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_t)

    layer_name = 'conv2d_1'
    layer_fm = activations[1]

    fig = showAllChannelsInFeatureMap(layer_name, layer_fm)

#initTest()
inceptionModel = Inception('inceptionv3_flowers.h5')

def convertImgToFig(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    mp_fig = mpld3.fig_to_html(fig)
    return mp_fig

######################################### Socket #############################################
@socketio.on('init')
def init():
    emit("layer_names", inceptionModel.layers)
    inceptionModel.currIdxs = []

@socketio.on('changeImg')
def changeImg(msg):
    imgName = msg['name']
    if inceptionModel.currImg != imgName:
        print("Get activations for new image " + imgName)
        inceptionModel.currImg = imgName
        inceptionModel.getModelActivations()
        emit('gotfigs', getFigsArrayByIdxs())

@socketio.on('allFm')
def allFm(layerIdx):
    print("emit all FM ", layerIdx)
    if isinstance(layerIdx, int):
        if layerIdx not in inceptionModel.currIdxs:
            inceptionModel.currIdxs.append(layerIdx)
            print(len(inceptionModel.currIdxs))
        emit('gotfig', getFigForLayer(layerIdx))

@socketio.on('filteredFm')
def filteredFm(msg):
    if msg['filter']:
        if msg['filter'] == "activations":
            figs = getFigsArrayByFilter(msg['filter'])
            emit('gotfigs', figs)
######################################################################################
if __name__ == "__main__":
    # format = "%(asctime)s: %(message)s"
    # logging.basicConfig(format=format, level=logging.INFO,
    #                     datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    # app.run(debug=True, use_reloader=False)
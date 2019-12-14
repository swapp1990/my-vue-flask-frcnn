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
from matplotlib.collections import EventCollection
import statistics

import lucid.modelzoo.vision_models as models

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit
# import eventlet
# eventlet.monkey_patch()

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
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

DATASET_DIR = "C:\\Users\Swapinl\Documents\Datasets\\flowers"
TRAIN_DIR = DATASET_DIR +  "\\train"
VALID_DIR = DATASET_DIR + "\\validation"
IMG_SIZE = (299, 299, 3)
BATCH_SIZE = 32

active_queues = []
class Worker(threading.Thread):
    def __init__(self, id):
        threading.Thread.__init__(self)
        self.mailbox = queue.Queue()
        self.id = id
        print("thread ", self.id)
        active_queues.append(self.mailbox)
    
    def run(self):
        while True:
            data = self.mailbox.get()
            print(self, 'received a message', data['action'], str(data['id']))
            if self.id == data['id']:
                if data['action'] == 'start':
                    self.doWork()
                if data['action'] == 'logs':
                    self.emitLogs(data['logs'])
                if data['action'] == 'showFig':
                    self.emitFig(data['fig'])
                if data['action'] == 'showGraph':
                    self.emitGraph(data['fig'])

    def doWork(self):
        inceptionModel.trainModel()
    
    def emitLogs(self, logs):
        print("emitLogs ", logs)
        obj = {"id": self.id, "loss": logs['loss'].item(), "batch":logs['batch']}
        socketio.emit('TrainingLogs', obj)
    
    def emitFig(self, fig):
        obj = {"id": self.id, "fig": fig}
        print("emitFig")
        socketio.emit('TrainingFigs', obj)
    
    def emitGraph(self, fig):
        obj = {"id": self.id, "fig": fig}
        print("emitGraph")
        socketio.emit('TrainingLossGraph', obj)

    def stop(self):
        self.mailbox.put("shutdown")
        self.join()

def broadcast_event(data):
    for q in active_queues:
        q.put(data)

class LossCallback(keras.callbacks.Callback):
    def __init__(self, model, img_t):
        self.model = model
        self.img_t = img_t
        self.layer_out = self.model.get_layer("activation_9").output
        self.lossesHist = []
        self.meanLossesHist = []
    
    def on_batch_end(self, batch, logs={}):
        print("")
        print("Logs ", logs)
        l = float("{0:.2f}".format(logs['loss']))
        #print(l)
        self.lossesHist.append(l)
        currMean = statistics.mean(self.lossesHist)
        self.meanLossesHist.append(currMean)

        lossGraph = None
        if len(self.lossesHist) > 2:
            lossGraph = self.getLossPlotFig(self.lossesHist, self.meanLossesHist)

        broadcastLogs(logs)
        if lossGraph is not None:
            broadcastLossGraph(lossGraph)
        
        if batch%5 == 0:
            activation_model = Model(inputs=self.model.input, outputs=self.layer_out)
            activation = activation_model.predict(self.img_t)
            fig = showAllChannelsInFeatureMap("activation_9", activation)
            broadcastFig(fig)
    
    def getLossPlotFig(self,data1,data2):
        xdata = [i for i in range(len(data1))]
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.plot(xdata, data1, 'b-', xdata, data2, 'r-')
        #plt.show()
        # fig = mpld3.fig_to_dict(fig)
        fig = mpld3.fig_to_html(fig)
        return fig

class Inception():
    def __init__(self):
        self.model = None
        self.currImg = 'daisy'
        self.activations = {}
        self.layers = []
        self.currIdxs = []
        train_datagen_f = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, rescale=1./255)
        val_datagen_f = ImageDataGenerator(rescale=1./255)
        self.train_gen = train_datagen_f.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode="categorical")
        self.val_gen = val_datagen_f.flow_from_directory(VALID_DIR, target_size=(IMG_SIZE[0], IMG_SIZE[1]), batch_size=BATCH_SIZE, class_mode="categorical")
    
    def trainModel(self):
        inp = Input(IMG_SIZE)
        inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=inp, input_shape=IMG_SIZE, pooling='avg')
        x = inception.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        out = Dense(5, activation='softmax')(x)

        full_model = Model(inp, out)
        #print(full_model.summary())
        full_model.compile(optimizer='adam', loss='categorical_crossentropy')

        imgT = self.getSampleImgTensor()
        lossCallback = LossCallback(full_model, imgT)
        full_model.fit_generator(self.train_gen, steps_per_epoch=100, epochs=5, validation_data=self.val_gen, verbose=1, callbacks=[lossCallback])
    
    def getSampleImgTensor(self):
        test_img = 'samples\\'+'daisy.jpg'
        img = image.load_img(test_img, target_size=(IMG_SIZE[0], IMG_SIZE[1]))
        img_t = image.img_to_array(img)
        img_t = np.expand_dims(img_t, axis=0)
        img_t /= 255.
        return img_t

    def loadModel(self, filename):
        self.model = load_model(filename)
    
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

def broadcastLogs(l):
    msg = {"action": "logs", "id": 0, "logs": l}
    broadcast_event(msg)

def broadcastFig(fig):
    msg = {"action": "showFig", "id": 0, "fig": fig}
    broadcast_event(msg)

def broadcastLossGraph(fig):
    msg = {"action": "showGraph", "id": 0, "fig": fig}
    broadcast_event(msg)

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

def initActivations():
    inceptionModel = Inception()
    inceptionModel.loadModel('inceptionv3_flowers.h5')
    inceptionModel.getModelActivations()
    return inceptionModel

def initTest():
    inceptionModel = Inception()
    inceptionModel.trainModel()
    return

#initTest()
#inceptionModel = initActivations()
inceptionModel = Inception()


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

@socketio.on('beginTraining')
def beginTraining():
    #inceptionModel.trainModel()
    thread = Worker(0)
    thread.start()
    thread2 = Worker(1)
    thread2.start()
    # msg = {"action": "start", "id": 0}
    # broadcast_event(msg)

@socketio.on('sendMsg')
def sendMsg():
    msg = {"action": "start", "id": 1}
    broadcast_event(msg)

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
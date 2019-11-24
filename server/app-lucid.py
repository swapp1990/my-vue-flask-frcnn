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
from scipy.misc import imread, imresize, imsave
import imageio

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
import IPython.display

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

from swaplucid import swapRender, interpretRender
import global_vars as G

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
#socketio = SocketIO(app, cors_allowed_origins="*")
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def convertImgToFig(img, style=None):
    figHeight = 2
    figWidth = 2
    if style is not None:
        figHeight = int(style['height']/100)
        figWidth = int(style['width']/100)
    plt.axis('off')
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    #(200, 200) size
    #print(figWidth, figHeight)
    plt.gcf().set_size_inches(figWidth, figHeight)
    #plt.show()
    mp_fig = mpld3.fig_to_html(fig)
    return mp_fig

######################################### Socket #############################################
#Init the type of generator for image viz.
def initGenerator(layer_name, filter_idx, config):
    print("config ", config)
    model = models.InceptionV1()
    model.load_graphdef()
    #print("Model loaded")

    show_negative = config['negative']
    n_batch = config['batch']
    img_gen = None
    #The original 'render_vis'
    if config['diversity']:
        #Generator for deversity objective
        img_gen = swapRender.diversity_render_yield(model, filter_idx, layer_name, n_batch=n_batch)
    else:
        #Generator for default objective
        regul_config = None
        if 'regul' in config:
            regul_config = config['regul']
        img_gen = swapRender.render_vis_yield(model, layer_name, filter_idx, show_negative=show_negative, use_regularizer=config['use_regularizer'], regularizer_params=regul_config, max_steps=2000)
    return img_gen

def initGenerator2(neurons, obj_op, config):
    model = models.InceptionV1()
    model.load_graphdef()
    n_batch = config['batch']
    img_gen = swapRender.render_vis_neurons_yield(model, neurons, max_steps=500)
    return img_gen

@socketio.on('connect')
def connect():
    print("reset")
    for t in G.active_threads:
        t.stop()
    G.active_threads.clear()

@socketio.on('startLucid')
def startLucid(filter_idx=0):
    model = models.InceptionV1()
    model.load_graphdef()
    print("Model loaded")

    #swapRender.render_vis(model, "mixed4a_pre_relu:476")
    layer_name = "mixed4a_pre_relu"
    name = layer_name + ":" + str(filter_idx)
    print(name)
    G.img_gen = swapRender.render_vis_yield(model, name,filter_idx, max_steps=500)
    img = next(G.img_gen)
    emit('gotfig', convertImgToFig(img))

@socketio.on('nextImg')
def nextImg(step):
    print("Step ", step)
    if step < 498:
        img = next(G.img_gen)
        time.sleep(0.2)
        emit('gotfig', convertImgToFig(img))
######################################################################################
######################################### Thread #############################################
active_queues = []
def outsideThreadCallTest():
    print(outsideThreadCallTest)

class Worker(threading.Thread):
    def __init__(self, params):
        threading.Thread.__init__(self)
        self.mailbox = queue.Queue()
        self.id = params["id"]
        self.step = 0
        self.style = params["style"]
        self.config = params["config"]
        self.imgGenerator = None
        # For layer:idx
        self.layer_name = params["layer"]
        self.filter_idx = int(params["filter_idx"])
        inp = self.layer_name + ":" + str(params["filter_idx"])
        print("init worker "+ inp)
        self.imgGenerator = initGenerator(self.layer_name, self.filter_idx, self.config)

        # For 2 Neurons (layer:idx)
        # self.imgGenerator = initGenerator2(params["neurons"], params["obj_op"], self.config)
        active_queues.append(self.mailbox)
                    
    def doWork(self):
        # socketio.emit('test', self.id, broadcast=True)
        try:
            img = next(self.imgGenerator)
            fig = convertImgToFig(img, self.style)
            obj = {"id": self.id, "step": self.step, "fig": fig}
            socketio.emit('workFinished', obj)
            self.step = self.step+1
        except:
            print("Emit Generator Exception to Client")
            obj = {"id": self.id, "step": self.step, "exception": True}
            socketio.emit('workFinished', obj)

    def saveImg(self):
        img = next(self.imgGenerator)
        filename = self.layer_name + "_" + str(self.filter_idx) + "_" + str(img.shape[0])
        filename += ".png"
        imageio.imwrite('savedimgs/'+filename, img)

    def loadImg(self, params):
        currFilterIdx = params["filter_idx"]
        filename = self.layer_name + "_" + str(currFilterIdx) + "_" + str(128)
        filename += ".png"
        try:
            img = imageio.imread('savedimgs/'+filename, as_gray=False, pilmode="RGB")
            fig = convertImgToFig(img, self.style)
            obj = {"id": self.id, "step": self.step, "fig": fig, "saved": True}
            socketio.emit('workFinished', obj)
        except:
            obj = {"id": self.id, "notFound": True}
            socketio.emit('workFinished', obj)

    def modifyThread(self, params):
        print("modifed ", params) 
        self.filter_idx = int(params["filter_idx"])
        self.step = 0

        self.imgGenerator = initGenerator(self.layer_name, self.filter_idx, self.config)
        img = next(self.imgGenerator)
        fig = convertImgToFig(img, self.style)
        obj = {"id": self.id, "step": self.step, "fig": fig}
        socketio.emit('workFinished', obj)
        self.step = self.step+1

    def run(self):
        # outsideThreadCallTest()
        # socketio.emit('test', self.id)
        while True:
            data = self.mailbox.get()
            
            #print("data ", data)
            if 'action' in data:
                if data['action'] == 'stop':
                    print(self, 'shutting down')
                    return
                    # if self.id == data['id']:
                    #     socketio.emit('threadStopped', self.id)
                    #     return
            #print(self, 'received a message:', data['id'])
            if self.id == data['id']:
                if data['action'] == 'perform':
                    self.doWork()
                elif data['action'] == 'save':
                    self.saveImg()
                elif data['action'] == 'load':
                    self.loadImg(data["params"])
                elif data['action'] == 'modify':
                    self.modifyThread(data['modifiedParams'])
    
    def getId(self):
        return self.id

    def stop(self):
        # outsideThreadCallTest()
        # socketio.emit('test', self.id)
        print("stop thread", self.id)
        self.imgGenerator = None
        print(len(active_queues))
        if len(active_queues) > 0:
            active_queues.remove(self.mailbox)
            socketio.emit('threadFinished', self.id)
        self.mailbox.put({"action":"stop"})
        self.join()

def broadcast_event(data):
    for q in active_queues:
        q.put(data)

@socketio.on('startNewThread')
def startNewThread(params):
    #print(params)
    thread = Worker(params)
    thread.start()
    print("start thread " + str(params['id']))
    G.active_threads.append(thread)
    emit("threadStarted", params['id'])

@socketio.on('perform')
def perform(id):
    msg = {"id": id, "action": "perform"}
    broadcast_event(msg)

@socketio.on('modify')
def modify(params):
    print("modify")
    msg = {"id": params["id"], "action": "modify", "modifiedParams": params}
    broadcast_event(msg)

@socketio.on('stopThread')
def stopThread(id):
    for t in G.active_threads:
        if id == t.getId():
            #print("active_queues", len(active_queues))
            t.stop()

@socketio.on('save')
def save(id):
    msg = {"id": id, "action": "save"}
    broadcast_event(msg)

@socketio.on('load')
def load(params):
    print('Load saved img')
    msg = {"id": params["id"], "action": "load", "params": params}
    broadcast_event(msg)

######################################################################################

def initTest():
    print('init')
    model = models.InceptionV1()
    model.load_graphdef()
    interpretRender.channel_attr()
    #swapRender.test_obj_f(model)
    # swapRender.render_vis_test(model)
    # imgGenerator = initGenerator(42)
    # img = next(imgGenerator)
    # print(img.shape)
    # convertImgToFig(img)
    #initGraph()

#initTest()
if __name__ == "__main__":
    # format = "%(asctime)s: %(message)s"
    # logging.basicConfig(format=format, level=logging.INFO,
    #                     datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    #app.run(debug=True, use_reloader=False)
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

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

from swaplucid import swapRender
import global_vars as G

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, async_mode='threading')
#socketio = SocketIO(app, cors_allowed_origins="*")
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

def initTest():
    model = models.InceptionV1()
    model.load_graphdef()
    _ = swapRender.render_vis(model, "mixed4a_pre_relu:476")

#initTest()

def convertImgToFig(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    mp_fig = mpld3.fig_to_html(fig)
    return mp_fig

######################################### Socket #############################################
def initGenerator(filter_idx):
    model = models.InceptionV1()
    model.load_graphdef()
    print("Model loaded")

    layer_name = "mixed4a_pre_relu"
    name = layer_name + ":" + str(filter_idx)
    print(name)
    img_gen = swapRender.render_vis_yield(model, name, max_steps=500)
    return img_gen

@socketio.on('startLucid')
def startLucid(filter_idx=0):
    model = models.InceptionV1()
    model.load_graphdef()
    print("Model loaded")

    #swapRender.render_vis(model, "mixed4a_pre_relu:476")
    layer_name = "mixed4a_pre_relu"
    name = layer_name + ":" + str(filter_idx)
    print(name)
    G.img_gen = swapRender.render_vis_yield(model, name, max_steps=500)
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
        self.filter_idx = params["filter_idx"]
        self.imgGenerator = initGenerator(self.filter_idx)
        active_queues.append(self.mailbox)
                    
    def doWork(self):
        # socketio.emit('test', self.id, broadcast=True)
        img = next(self.imgGenerator)
        #print(img.shape)
        fig = convertImgToFig(img)
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
                self.doWork()
    
    def getId(self):
        return self.id

    def stop(self):
        outsideThreadCallTest()
        socketio.emit('test', self.id)
        print("stop thread", self.id)
        active_queues.remove(self.mailbox)
        socketio.emit('threadFinished', self.id)
        self.mailbox.put({"action":"stop"})
        self.join()

def broadcast_event(data):
    for q in active_queues:
        q.put(data)

@socketio.on('startNewThread')
def startNewThread(params):
    print(params)
    thread = Worker({"id": params['id'], "filter_idx": params['filterIndex']})
    thread.start()
    print("start thread " + str(params['id']) + " filter " + str(params['filterIndex']))
    G.active_threads.append(thread)

@socketio.on('perform')
def perform(id):
    msg = {"id": id, "action": "perform"}
    broadcast_event(msg)

@socketio.on('stopThread')
def stopThread(id):
    for t in G.active_threads:
        if id == t.getId():
            #print("active_queues", len(active_queues))
            t.stop()
######################################################################################

if __name__ == "__main__":
    # format = "%(asctime)s: %(message)s"
    # logging.basicConfig(format=format, level=logging.INFO,
    #                     datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    # app.run(debug=True, use_reloader=False)
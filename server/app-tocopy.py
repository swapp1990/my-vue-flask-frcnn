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
socketio = SocketIO(app, cors_allowed_origins="*")
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
@socketio.on('startLucid')
def startLucid():
    model = models.InceptionV1()
    model.load_graphdef()
    print("Model loaded")

    #swapRender.render_vis(model, "mixed4a_pre_relu:476")
    G.img_gen = swapRender.render_vis_yield(model, "mixed4a_pre_relu:476")
    img = next(G.img_gen)
    emit('gotfig', convertImgToFig(img))
######################################################################################
if __name__ == "__main__":
    # format = "%(asctime)s: %(message)s"
    # logging.basicConfig(format=format, level=logging.INFO,
    #                     datefmt="%H:%M:%S")

    print("running socketio")
    socketio.run(app)
    # app.run(debug=True, use_reloader=False)
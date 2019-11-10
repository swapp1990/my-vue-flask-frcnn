from keras import backend as K
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

import logging
import threading
import time
import json
import concurrent.futures

from keras.applications import VGG16
from keras import activations
import global_vars as G
import eventlet
eventlet.monkey_patch()

# Module to init
from mymodules import CNNInterViz as viz

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

#viz.__init__()

def emitFinished(step):
    #socketio.emit('threadFinished', dict(foo='bar'))
    socketio.emit('threadFinished', step)

def listen(step):
    print("Thread %s: starting", step)
    eventlet.sleep(5)
    emitFinished(step)
    print("Thread %s: finishing", step)

@socketio.on('startNewThread')
def startNewThread(step):
    #executor.submit(thread_function, step)
    eventlet.spawn(listen, step)

if __name__ == '__main__':
    #app.run(debug=True, use_reloader=True)
    print("running socketio")
    socketio.run(app)


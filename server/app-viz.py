from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

from keras.applications import VGG16
from keras import activations
from util import viz_utils
from vis import activation_maximization as am
from matplotlib import pyplot as plt
import mpld3
from mpld3 import plugins
import time
# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

#app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}}) 

global opt
opt = None
#init()

@socketio.on('first-connect1')
def handle_message():
    #print('received message: ' + message)
    print('first connect')

@socketio.on('firstclick')
def firstclickRes(class_idx):
    #print('received message: ' + message)
    print('clicked')
    model_imagenet = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    # Swap softmax with linear
    model_imagenet.layers[layer_idx].activation = activations.linear
    model_imagenet = viz_utils.apply_modifications(model_imagenet)
    global opt
    opt = am.initOptimizer(model_imagenet, layer_idx, class_indices=class_idx)
    emit('optinit')
    # plt.imshow(img)
    # plt.show() 
    #emit('pong')

@socketio.on('performminimize')
def performMinimize(step):
    print("performMinimize ", step)
    if step < 300:
        if opt is not None:
            img = am.visualize_activation_single(opt)
            fig, ax = plt.subplots()
            ax.imshow(img)
            fig = mpld3.fig_to_html(fig)
            emit('gotfig', fig)
        else:
            print("opt error")

@socketio.on('ping')
def pongResponse():
    print("ping")
    emit('pong')

if __name__ == '__main__':
    #app.run(debug=True, use_reloader=True)
    print("running socketio")
    socketio.run(app)
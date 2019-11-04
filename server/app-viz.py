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

global opt, model_imagenet, g_cls_idx
opt = None
model_imagenet = None
g_cls_idx = 20

# def init():
#     global fig, ax

def testinit():
    model_imagenet = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    model_imagenet.layers[layer_idx].activation = activations.linear
    model_imagenet = viz_utils.apply_modifications(model_imagenet)
    img = am.visualize_activation(model_imagenet, layer_idx, class_indices=20)

#init()
#testinit()


@socketio.on('first-connect1')
def handle_message():
    #print('received message: ' + message)
    print('first connect')

@socketio.on('firstclick')
def firstclickRes(class_idx):
    #print('received message: ' + message)
    print('clicked')
    global model_imagenet, g_cls_idx
    model_imagenet = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    # Swap softmax with linear
    model_imagenet.layers[layer_idx].activation = activations.linear
    model_imagenet = viz_utils.apply_modifications(model_imagenet)
    global opt
    opt = am.initOptimizer(model_imagenet, layer_idx, class_indices=class_idx)
    g_cls_idx = class_idx
    emit('optinit')

@socketio.on('reset')
def reset():
    print("reset ", g_cls_idx)
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    am.modifyOpt(opt, model_imagenet, layer_idx, class_indices=g_cls_idx, l2_norm=True)

@socketio.on('modify')
def modify(params):
    print("modify ", params)
    g_cls_idx = params['cls_idx']
    l2_norm_bool = params['l2_norm']
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    am.modifyOpt(opt, model_imagenet, layer_idx, class_indices=g_cls_idx, l2_norm=l2_norm_bool)

@socketio.on('performminimize')
def performMinimize(step):
    print("performMinimize ", step)
    #global fig
    if step < 300:
        if opt is not None:
            fig, ax = plt.subplots()
            #lines = ax.plot(range(10), 'o')
            img = am.visualize_activation_single(opt)
            ax.imshow(img)
            #points = ax.plot(range(10), 'o')
            #labels = ['<h1>{title}</h1>'.format(title=i) for i in range(10)]
            # plugins.connect(fig, plugins.PointLabelTooltip(points[0]))
            #plugins.connect(fig, plugins.PointHTMLTooltip(points[0], labels))
            mp_fig = mpld3.fig_to_html(fig)
            emit('gotfig', mp_fig)
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
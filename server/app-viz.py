from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

from keras.applications import VGG16
from keras import activations
from util import viz_utils
from vis import activation_maximization as am
from vis import saliency
from matplotlib import pyplot as plt
import numpy as np
import mpld3
from mpld3 import plugins
import time
import matplotlib.cm as cm
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
def overlay(array1, array2, alpha=0.5):
    """Overlays `array1` onto `array2` with `alpha` blending.

    Args:
        array1: The first numpy array.
        array2: The second numpy array.
        alpha: The alpha value of `array1` as overlayed onto `array2`. This value needs to be between [0, 1],
            with 0 being `array2` only to 1 being `array1` only (Default value = 0.5).

    Returns:
        The `array1`, overlayed with `array2` using `alpha` blending.
    """
    if alpha < 0. or alpha > 1.:
        raise ValueError("`alpha` needs to be between [0, 1]")
    if array1.shape != array2.shape:
        raise ValueError('`array1` and `array2` must have the same shapes')

    return (array1 * alpha + array2 * (1. - alpha)).astype(array1.dtype)
    
def testinit():
    model_imagenet = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model_imagenet, 'predictions')
    #filters = np.arange(viz_utils.get_num_filters(model_imagenet.layers[layer_idx]))

    model_imagenet.layers[layer_idx].activation = activations.linear
    model_imagenet = viz_utils.apply_modifications(model_imagenet)
    #img = am.visualize_activation(model_imagenet, layer_idx, class_indices=0)
    img1 = viz_utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
    img2 = viz_utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

    f, ax = plt.subplots(1, 2)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = saliency.visualize_cam(model_imagenet, layer_idx, class_indices=20, seed_input=img)
        #grads = saliency.visualize_saliency(model_imagenet, layer_idx, class_indices=20, seed_input=img)
        #ax[i].imshow(grads, cmap='jet')
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i].imshow(overlay(jet_heatmap, img))
    plt.show()

#init()
testinit()


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
    #layer_idx = viz_utils.find_layer_idx(model_imagenet, 'block1_conv2')
    
    # Swap softmax with linear
    #model_imagenet.layers[layer_idx].activation = activations.linear
    #model_imagenet = viz_utils.apply_modifications(model_imagenet)
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
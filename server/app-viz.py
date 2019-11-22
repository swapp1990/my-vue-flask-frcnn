import base64
from matplotlib import pyplot as plt
import numpy as np
import mpld3
import json
from mpld3 import plugins
import time
import matplotlib.cm as cm
from scipy.misc import imread, imresize, imsave
import re
import imageio
import tensorflow as tf
from keras import backend as K
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response
from flask_socketio import SocketIO, emit

from keras.applications import VGG16
from keras import activations
from util import viz_utils
from vis import activation_maximization as am
from vis import saliency
import global_vars as G

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

####################################################################################################
#Save uploaded image as a global var
def convertImage(imgData):
    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('client_img.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
        print('Saved as client_img.png')

@app.route('/api/upload', methods=['GET','POST'])
# @cross_origin(allow_headers=['Content-Type'])
def upload_img():
    if request.method == 'POST':
        """ Receive base 64 encoded image """
        print('Image received')
        imgData = request.get_data()
        convertImage(imgData)
        # read the image into memory and convert into numpy array
        x = imageio.imread('client_img.png', as_gray=False, pilmode="RGB")
        G.client_img = x
        print('client_img ', G.client_img.shape)
        return "Success"
####################################################################################################
######################################### Saliency #############################################
def initImagenet():
    model = VGG16(weights='imagenet', include_top=True)
    layer_idx = viz_utils.find_layer_idx(model, 'predictions')
    model.layers[layer_idx].activation = activations.linear
    model = viz_utils.apply_modifications(model)
    return model, layer_idx

@app.route('/getSaliency', methods=['GET','POST'])
def getSaliency():
    print('getSaliency')
    plt.figure(figsize=(8,8))
    fig, ax = plt.subplots()
    model, layer_idx = initImagenet()
    img = viz_utils.load_img('client_img.png', target_size=(224, 224))
    grads = saliency.visualize_saliency(model, layer_idx, class_indices=235, seed_input=img)
    ax.imshow(grads, cmap='jet')
    mp_fig = mpld3.fig_to_html(fig)
    return Response(json.dumps([mp_fig]),  mimetype='application/json')

@app.route('/getGradCam', methods=['GET','POST'])
def getGradCam():
    print('getGradCam')
    plt.figure(figsize=(8,8))
    fig, ax = plt.subplots()
    model, layer_idx = initImagenet()
    img = viz_utils.load_img('client_img.png', target_size=(224, 224))
    grads = saliency.visualize_cam(model, layer_idx, class_indices=235, seed_input=img)
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    ax.imshow(overlay(jet_heatmap, img))
    mp_fig = mpld3.fig_to_html(fig)
    return Response(json.dumps([mp_fig]),  mimetype='application/json')
####################################################################################################

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

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def testinit_conv():
    model = VGG16(weights='imagenet', include_top=True)
    layer_name = 'block3_conv1'
    filter_index = 15
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    img = deprocess_image(img)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

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
#testinit()
#testinit_conv()

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
    #layer_idx = viz_utils.find_layer_idx(model_imagenet, 'block5_conv3')

    global opt
    opt = am.initOptimizer(model_imagenet, layer_idx, class_indices=56)
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
    if step < 100:
        if opt is not None:
            fig, ax = plt.subplots()
            #lines = ax.plot(range(10), 'o')
            time.sleep(1)
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
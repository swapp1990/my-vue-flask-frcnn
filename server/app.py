import uuid
import tensorflow as tf
from scipy.misc import imread, imresize
from PIL import Image
import numpy as np
import io
import re
import sys
import base64
import os
import cv2
import json
import base64

from flask import Flask, jsonify, request
from flask_cors import CORS

from keras.models import model_from_json
from flask import send_file

import frcnn

sys.path.append(os.path.abspath("./model"))

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

global model, graph

img2 = None

def init():
    model_base = frcnn.getBaseLayers()
    #model_base.load_weights('./model_frcnn.hdf5', by_name=True)
    # print(model_base.summary())
    return model_base

init()
graph = tf.get_default_graph()
im_size = 600
img_channel_mean = [103.939, 116.779, 123.68]
img_scaling_factor = 1.0

def convertImage(imgData):
    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

def format_img_size(img):
    """ formats the image size based on config """
    img_min_side = float(im_size)
    (height, width ,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= img_channel_mean[0]
    img[:, :, 1] -= img_channel_mean[1]
    img[:, :, 2] -= img_channel_mean[2]
    img /= img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def format_img(img):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img)
    img = format_img_channels(img)
    return img, ratio

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

@app.route('/predict', methods=['POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
     # read the image into memory
    x = imread('output.png', mode='L')
    x = imresize(x, (28,28))/255
    x = x.reshape(1,28,28,1)

    with graph.as_default():
        pred = model.predict(x)
    prob = pred[0][pred.argmax()] * 100
    prob = round(prob, 2)
    mypred = pred.argmax()
    return jsonify(pred=int(mypred), prob=int(prob))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        print('Get Request received')
        with open("cv2.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string

    if request.method == 'POST':
        print('Request received')
        request_data = json.loads(request.get_data())
        data = request_data['data'][5:]

        with open('file.img', 'w') as wf:
            wf.write(data)
            
        print('Saved in file.')

        with open('file.img', 'r') as rf:
            data = rf.read()
            mimetype, image_string = data.split(';base64,')
            image_bytes = image_string.encode('utf-8')
            img = base64.decodebytes(image_bytes)
            pilImg = Image.open(io.BytesIO(img))
            img2 = cv2.cvtColor(np.array(pilImg), cv2.COLOR_BGR2RGB)
            
            X, ratio = format_img(img2)
            X = np.transpose(X, (0, 2, 3, 1))

            cv2.imwrite('cv2.png',X)
        return 'detect'

if __name__ == '__main__':
    app.run()
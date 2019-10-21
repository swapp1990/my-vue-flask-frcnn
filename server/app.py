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
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

from keras.models import model_from_json
from keras import backend as K
from flask import send_file
from matplotlib import pyplot as plt

import frcnn
import roi_helpers

sys.path.append(os.path.abspath("./model"))

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

global model_rpn, model_classifier, graph

img2 = None

graph = tf.get_default_graph()
im_size = 600
img_channel_mean = [103.939, 116.779, 123.68]
img_scaling_factor = 1.0
rpn_stride = 16
num_rois = 32

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

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

def process_frcnn(img):
    bbox_threshold = 0.8
    class_mapping = {'tvmonitor': 0, 'train': 1, 'person': 2, 'boat': 3, 'horse': 4, 'cow': 5, 'bottle': 6, 'dog': 7, 'aeroplane': 8, 'car': 9, 'bus': 10, 'bicycle': 11, 'chair': 12, 'diningtable': 13, 'pottedplant': 14, 'bird': 15, 'cat': 16, 'motorbike': 17, 'sheep': 18, 'sofa': 19, 'bg': 20}

    class_mapping = {v: k for k, v in class_mapping.items()}
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

    X, ratio = format_img(img)
    X = np.transpose(X, (0, 2, 3, 1))
    with graph.as_default():
        [Y1, Y2, F] = model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, K.common.image_dim_ordering(), overlap_thresh=0.7)
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//num_rois + 1):
            ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == 9:
                continue

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold:
                    continue
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= classifier_regr_std[0]
                        ty /= classifier_regr_std[1]
                        tw /= classifier_regr_std[2]
                        th /= classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([rpn_stride*x, rpn_stride*y, rpn_stride*(x+w), rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))
        
        displayRects = []
        texts = []
        for key in bboxes:
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                print(textLabel)
                if key != 'bg':
                    displayRects.append([real_x1, real_y1, real_x2, real_y2])
                    texts.append(textLabel)
                    
        print("Rects found ", len(displayRects))
        # displayBoxes(img, anchors=displayRects, texts=texts)
        processed_img = saveResult(img, anchors=displayRects, texts=texts)
        return processed_img

def init():
    model_rpn, model_classifier = frcnn.getCompiledModel()
    return model_rpn, model_classifier
    
def initTest():
    img = cv2.imread('images/persons.jpg')
    processed_img = process_frcnn(img)
    cv2.imwrite('result/cv2.png',processed_img)

def detectAndSave(img):
    processed_img = process_frcnn(img)
    cv2.waitKey(0)
    cv2.imwrite('result/cv2.png',processed_img)
    print("Image saved")

def saveResult(img, anchors=[], texts=[]):
    coloraa = (255, 0, 0)
    for i in range(len(anchors)):
        a = anchors[i]
        txt = texts[i]
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        cv2.putText(img, txt, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
        cv2.rectangle(img, (x1, y1), (x2, y2), coloraa, 2)
        cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, coloraa, -1)
    return img

def displayBoxes(img, anchors=[], rois=[], showBox=True, texts=[]): 
    colorbb = (0, 255, 0)
    coloraa = (255, 0, 0)
    
    for i in range(len(anchors)):
        a = anchors[i]
        txt = texts[i]
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        cv2.putText(img, txt, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 

        if showBox:
            cv2.rectangle(img, (x1, y1), (x2, y2), coloraa, 2)
        cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, coloraa, -1)
    
    for i in range(len(rois)):
        roi = rois[i]
        x1, y1, x2, y2 = int(roi[0]*rpn_stride), int(roi[2]*rpn_stride), int(roi[1]*rpn_stride), int(roi[3]*rpn_stride)
        #print(x1, y1, x2, y2)
        cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, coloraa, -1)
    
    # for i in range(len(regrs)):
    #     regr = regrs[i]
    #     cx1, cx2 = int(regr[0]*16), int(regr[1]*16)
    #     print(cx1, cx2)
    #     cv2.circle(img, (cx1, cx2), 3, coloraa, -1)
    
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.show()

model_rpn, model_classifier = init()
#initTest()

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

# @app.route('/predict', methods=['POST'])
# def predict():
#     imgData = request.get_data()
#     convertImage(imgData)
#      # read the image into memory
#     x = imread('output.png', mode='L')
#     x = imresize(x, (28,28))/255
#     x = x.reshape(1,28,28,1)

#     with graph.as_default():
#         pred = model_rpn.predict(x)
#     prob = pred[0][pred.argmax()] * 100
#     prob = round(prob, 2)
#     mypred = pred.argmax()
#     return jsonify(pred=int(mypred), prob=int(prob))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        print('Get Request received')
        with open("result/cv2.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string

    if request.method == 'POST':
        print('Request received')
        request_data = json.loads(request.get_data())
        data = request_data['data'][5:]

        with open('result/file.img', 'w') as wf:
            wf.write(data)
            
        print('Saved Original in file.')

        with open('result/file.img', 'r') as rf:
            data = rf.read()
            mimetype, image_string = data.split(';base64,')
            image_bytes = image_string.encode('utf-8')
            img = base64.decodebytes(image_bytes)
            pilImg = Image.open(io.BytesIO(img))
            img2 = cv2.cvtColor(np.array(pilImg), cv2.COLOR_BGR2RGB)
            detectAndSave(img2)
        return 'detect'

if __name__ == '__main__':
    app.run()
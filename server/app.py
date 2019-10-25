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
import random

from flask import Flask, jsonify, request
from flask_cors import CORS

from keras.models import model_from_json
from keras import backend as K
from flask import send_file

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import frcnn
import roi_helpers

from util import draw_plots
import global_vars as g

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
num_rois = 32

#Load models and trained weights
def init():
    model_rpn, model_classifier = frcnn.getCompiledModel()
    print("global models set")
    return model_rpn, model_classifier

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

    g.class_mapping = {v: k for k, v in g.class_mapping.items()}
    
    X, ratio = format_img(img)

    X = np.transpose(X, (0, 2, 3, 1))
    with graph.as_default():
        #Proposes regions on the image X, for all anchor and points on the image, gives a sigmoid class
        # and a regr value
        [Y1, Y2, F] = model_rpn.predict(X)
        #step1(Y1)

        #Select the top 300 RPN and get the ROI (x1,y1,x2,y2) based on overlap threshold.
        R = roi_helpers.rpn_to_roi(Y1, Y2, K.common.image_dim_ordering(), overlap_thresh=0.7)
        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        debug_bboxes1 = []
        debug_bboxes2 = []
        #print(R.shape[0]) #300//32 => 9 pools with 32 ROI
        for jk in range(R.shape[0]//num_rois + 1):
            ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            
            #If the last pool has less ROIs, just add same ROIs as a padding to proper form the shape
            if jk == R.shape[0]//num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            #For 32 ROIs, predict probability (all 21 classes) and regr boxes (matches the class?)
            #P_regr: 4 coord of Regression for each class (out of 20). 20 classes minus 'bg'
            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            #print(P_cls.shape) #(1,32,21) (1,num_rois, num_classes)
            #print(P_regr.shape) #(1,32,80) (1,num_rois,4*(num_classes-1))
            debug_bboxes1 = []
            debug_bboxes2 = []
            debug_bboxes3 = []
            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold:
                    (x, y, w, h) = ROIs[0, ii, :]
                    debug_bboxes1.append([g.rpn_stride*x, g.rpn_stride*y, g.rpn_stride*(x+w), g.rpn_stride*(y+h)])
                    continue
                cls_name = g.class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]
                debug_bboxes2.append([g.rpn_stride*x, g.rpn_stride*y, g.rpn_stride*(x+w), g.rpn_stride*(y+h)])

                max_cls_idx = np.argmax(P_cls[0, ii, :])
                #Every cls has correspongin 4 coords of reg. So we select reg of max cls only from P_regr
                try:
                    #Get Regr of only the selected Class using cls_num
                    (tx, ty, tw, th) = P_regr[0, ii, 4*max_cls_idx:4*(max_cls_idx+1)]
                    #classifier_regr_std gives the proper scale to apply regr.
                    tx /= g.classifier_regr_std[0]
                    ty /= g.classifier_regr_std[1]
                    tw /= g.classifier_regr_std[2]
                    th /= g.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    debug_bboxes3.append([g.rpn_stride*x, g.rpn_stride*y, g.rpn_stride*(x+w), g.rpn_stride*(y+h)])

                except:
                    pass
                bboxes[cls_name].append([g.rpn_stride*x, g.rpn_stride*y, g.rpn_stride*(x+w), g.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
            #step1(img,jk,debug_bboxes1,debug_bboxes2,debug_bboxes3,ratio)

        displayRects = []
        texts = []
        for key in bboxes:
            bbox_arr = np.array(bboxes[key])
            prob_arr = np.array(probs[key])
            #step2(img, bbox_arr,prob_arr,key, ratio)
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox_arr,prob_arr,overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                #print(textLabel)
                if key != 'bg':
                    displayRects.append([real_x1, real_y1, real_x2, real_y2])
                    texts.append(textLabel)
                    
        print("Rects found ", len(displayRects))
        fig = draw_plots.displayBoxes(img, anchors=displayRects, texts=texts)
        #processed_img = saveResult(img, anchors=displayRects, texts=texts)
        #return processed_img
        return fig

def resetImg():
    img = cv2.imread('images/persons.jpg')

def initTest():
    img = cv2.imread('images/persons.jpg')
    #make this img global for debugview tasks
    settings.myDebugList['img'] = img
    processed_img = process_frcnn(img)
    cv2.imwrite('result/final_res.png',processed_img)

def processImage(img):
    X, ratio = format_img(img)
    settings.myDebugList['ratio'] = ratio

def detectAndSave(img):
    processed_img = process_frcnn(img)
    cv2.waitKey(0)
    cv2.imwrite('result/cv2.png',processed_img)
    print("Image saved")

def showFilterDebug(bbox_arr, left_idxs, picked_idxs, filtered_idxs, img, key, ratio):
    img_copy = img.copy()
    x1 = bbox_arr[:, 0]
    y1 = bbox_arr[:, 1]
    x2 = bbox_arr[:, 2]
    y2 = bbox_arr[:, 3]

    for idx in range(len(left_idxs)):
        i = left_idxs[idx]
        addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(255,0,0))

    for idx in range(len(picked_idxs)):
        i = picked_idxs[idx]
        addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(0,255,0), thickness=2)

    for idx in range(len(filtered_idxs)):
        i = filtered_idxs[idx]
        addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(0,0,255))

    n_picked = str(len(picked_idxs))
    if key != 'bg':
        filename = str('filter_step_'+key+'_'+n_picked+'.png')
        cv2.imwrite(str('result/'+filename),img_copy)
        #print('Removed ', len(filtered_idxs))
        print('Image saved ', filename)

def filterOverlapping(bbox_arr, probs_arr, img, key, ratio):
    img_copy = img.copy()
    # get area of all boxes
    x1 = bbox_arr[:, 0]
    y1 = bbox_arr[:, 1]
    x2 = bbox_arr[:, 2]
    y2 = bbox_arr[:, 3]
    area = (x2 - x1) * (y2 - y1)

    #Probs gives prob of each bbox, we sort the indexes which we use to get the corr. box
    sorted_idxs = np.argsort(probs_arr)
    picked_idxs = []
    overlap_thresh=0.5

    #Sort the bbox, picks the highest bbox, removes all bboxes who overlap (>0.5). Repeat
    while len(sorted_idxs) > 0:
        last = len(sorted_idxs) - 1
        i = sorted_idxs[last]
        picked_idxs.append(i)

        #Picks either the last box coord or the set of all other coords
        xx1_arr = np.maximum(x1[i], x1[sorted_idxs[:last]])
        yy1_arr = np.maximum(y1[i], y1[sorted_idxs[:last]])
        xx2_arr = np.minimum(x2[i], x2[sorted_idxs[:last]])
        yy2_arr = np.minimum(y2[i], y2[sorted_idxs[:last]])
        ww_int = np.maximum(0, xx2_arr - xx1_arr)
        hh_int = np.maximum(0, yy2_arr - yy1_arr)
        area_int = ww_int * hh_int
        area_union = area[i] + area[sorted_idxs[:last]] - area_int
        #Returns overlap value between curr bbox (top prob) and all other bbox
        overlap_arr = area_int/(area_union + 1e-6)
        #print("overlap_arr ", overlap_arr)
        #Filter overlapping boxes and remove them including selected box (last) from sorted_idxs
        filter_idxs = np.where(overlap_arr > overlap_thresh)[0]
        delete_idxs = np.concatenate(([last], filter_idxs))
        sorted_idxs = np.delete(sorted_idxs, delete_idxs)

        showFilterDebug(bbox_arr, sorted_idxs, picked_idxs, filter_idxs, img, key, ratio)
    
    #return the bbox which were picked
    bbox_arr = bbox_arr[picked_idxs].astype("int")
    probs_arr = probs_arr[picked_idxs].astype("int")

    return bbox_arr, probs_arr

#Spatial Pyramid Pooling
#Has 300 ROIs selected from the RPN trained model for the given image.
#Divide into batches of 32 ROIs and predict boxes and p_regr
#bbox_arr1: all boxes rejected (< threshold)
#bbox_arr2: all boxes selected
#bbox_arr3: all boxes after regr is applied
def step1(img, pool_idx, bbox_arr1, bbox_arr2, bbox_arr3, ratio):
    stepType = 'threshold_'
    filename = str('result/pyramidpooling/'+stepType+str(pool_idx)+'.png')
    save_with_mul_boxes(img, [bbox_arr1, bbox_arr2], ratio, filename)

    stepType = 'regr_'
    filename = str('result/pyramidpooling/'+stepType+str(pool_idx)+'.png')
    save_with_mul_boxes(img, [bbox_arr2, bbox_arr3], ratio, filename)

def step2(img, bbox_arr, probs_arr, key, ratio):
    save_images(img, bbox_arr, probs_arr, key, ratio, 'all')
    bbox_arr, probs_arr = filterOverlapping(bbox_arr, probs_arr, img, key, ratio)
    #save_images(img, bbox_arr, probs_arr, key, ratio, 'filtered')

################## Common func ###########################################
def display_images(img, bbox_arr1, bbox_arr2, ratio):
    img_copy = img.copy() #resets the cv2 part
    print(len(bbox_arr1), len(bbox_arr2))
    for i in range(len(bbox_arr1)):
        (x1, y1, x2, y2) = bbox_arr1[i]
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        addRectToImg(img_copy, x1, y1, x2, y2, color=(0,255,0))
    for i in range(len(bbox_arr2)):
        (x1, y1, x2, y2) = bbox_arr2[i]
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        addRectToImg(img_copy, x1, y1, x2, y2, color=(255,255,0))
    plt.imshow(img_copy)
    plt.show()

def save_with_mul_boxes(img, mul_cluster, ratio, filename):
    #Color of boxes for clusters (in order) !add more colors if more than 2 clusters
    colors=[(0,155,0), (255,0,0)]
    selColor = colors[0]
    img_copy = img.copy()
    for i in range(len(mul_cluster)):
        bbox_arr = mul_cluster[i]
        selColor = colors[i]
        for j in range(len(bbox_arr)):
            (x1, y1, x2, y2) = bbox_arr[j]
            (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            addRectToImg(img_copy, x1, y1, x2, y2, color=selColor)
    cv2.waitKey(0)
    cv2.imwrite(filename,img_copy)
    print('Image saved ', filename)

def save_images(img, bbox_arr, probs_arr, key, ratio, f_prefix):
    img_copy = img.copy() #resets the cv2 part
    for i in range(len(bbox_arr)):
        (x1, y1, x2, y2) = bbox_arr[i]
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        addRectToImg(img_copy, x1, y1, x2, y2)

    cv2.waitKey(0)
    filename = str(f_prefix+'_'+key+'.png')
    #plt.imshow(img_copy)
    #plt.show()
    cv2.imwrite(str('result/'+filename),img_copy)
    print('Image saved ', filename)

def addRectToImg(img, x1, y1, x2, y2, scaled=False, color=(255,0,0), thickness=1):
    if scaled:
        (x1, y1, x2, y2) = get_real_coordinates(2.0, x1, y1, x2, y2)

    #coloraa = (255, 0, 0)
    cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, color, 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
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

@app.route('/query', methods=['POST'])
def query():
    img = cv2.imread('images/persons.jpg')
    fig = process_frcnn(img)
    data = json.loads(request.data)
    return fig

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
    app.run(debug=True, use_reloader=True)
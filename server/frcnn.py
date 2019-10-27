import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from matplotlib import pyplot as plt
import asyncio

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from RoiPoolingConv import RoiPoolingConv

import global_vars as G
from util import image_processing as IP, draw_plots
import roi_helpers

num_features = 512
num_rois = 32

def nn_base(input_tensor=None, trainable=False):
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x

def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

def getBaseLayers():
    class_mapping = {'tvmonitor': 0, 'train': 1, 'person': 2, 'boat': 3, 'horse': 4, 'cow': 5, 'bottle': 6, 'dog': 7, 'aeroplane': 8, 'car': 9, 'bus': 10, 'bicycle': 11, 'chair': 12, 'diningtable': 13, 'pottedplant': 14, 'bird': 15, 'cat': 16, 'motorbike': 17, 'sheep': 18, 'sofa': 19, 'bg': 20}

    class_mapping = {v: k for k, v in class_mapping.items()}

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn_base(img_input, trainable=True)

    num_anchors = len(G.anchor_box_scales) * len(G.anchor_box_ratios)
    rpn_layers = rpn(shared_layers, num_anchors)
    classifier_layers = classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier_layers)

    return shared_layers, model_rpn, model_classifier

# def getRPNLayers(base_layers):
#     input_shape_img = (None, None, 3)
#     img_input = Input(shape=input_shape_img)
    
#     return rpn_layers

def getCompiledModel():
    shared_layers, model_rpn, model_classifier = getBaseLayers()
    #rpn_layers = getRPNLayers(shared_layers)
    #print(model_rpn.summary())
    model_rpn.load_weights('model/model_frcnn.hdf5', by_name=True)
    model_classifier.load_weights('model/model_frcnn.hdf5', by_name=True)
    print("Loaded weights")
    return model_rpn, model_classifier

#*******************   Process FRCNN Debug Inermediate *****************************#

#Inputs: 
# 'img': Input Image
def processRpnToROI(img):
    G.class_mapping = {v: k for k, v in G.class_mapping.items()}

    X, ratio = IP.format_img(img)
    G.ratio = ratio
    X = np.transpose(X, (0, 2, 3, 1))
    with G.graph.as_default():
        #Proposes regions on the image X, for all anchor and points on the image, gives a sigmoid class and a regr value.
        #cls_sigmoid => (1,37,60,9): (37,60) is the kernel size of the last base CNN layer. For each point in the kernel, for all 9 anchor points, gets the sigmoid value (0 to 1)
        #bbox_regr => (1,37,60,9*4): 4 coord for each anchor point
        #base_layers => (1,37,60,512): 512 is no of output filters at the end of chosen CNN layers.
        [cls_sigmoid, bbox_regr, base_layers] = G.model_rpn.predict(X)
        
        #Select the proper top 300 from all RPN values and get the ROI (x1,y1,x2,y2) based on overlap threshold.
        #R => (300, 4): 300 * (x1,y1,x2,y2)
        R = roi_helpers.rpn_to_roi(cls_sigmoid, bbox_regr, K.common.image_dim_ordering(), overlap_thresh=0.7, debug=True)
        G.ROIs = R
        print(G.ROIs.shape)

# def restartProcessRpn(img):
#     X, ratio = IP.format_img(img)
#     G.ratio = ratio
#     G.debug_img = img
#     X = np.transpose(X, (0, 2, 3, 1))
#     with G.graph.as_default():
#         [cls_sigmoid, bbox_regr, base_layers] = G.model_rpn.predict(X)
#         R = roi_helpers.rpn_to_roi(cls_sigmoid, bbox_regr, K.common.image_dim_ordering(), overlap_thresh=0.7, debug=True)
#         G.ROIs = R
#         fig = draw_plots.displayBoxes(G.debug_img, rois=Debug_R)
#         return fig
        
def getRpnToRoi(i_start=0, i_end=10, ratios=[], sizes=[]):
    Debug_R = G.ROIs[i_start:i_end,:]
    restart = False
    if len(sizes):
        sizesArr = [int(sizeStr) for sizeStr in sizes]
        if sizesArr != G.anchor_scales_d:
            G.anchor_scales_d = sizesArr
            restart = True
    if len(ratios):
        ratiosArr = []
        for ratio in ratios:
            if ratio == "1:1":
                ratiosArr.append([1,1])
            elif ratio == "1:2":
                ratiosArr.append([1,2])
            elif ratio == "2:1":
                ratiosArr.append([2,1])
        if ratiosArr != G.anchor_ratios_d:
            G.anchor_ratios_d = ratiosArr
            restart = True
    if restart:
        print(restart)
        processRpnToROI(G.debug_img)
    Debug_R = G.ROIs[i_start:i_end,:]
    #Debug: Display chosen ROIs on the image.
    print("display box")
    fig = draw_plots.displayBoxes(G.debug_img, rois=Debug_R)
    return fig

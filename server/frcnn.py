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
    roi_input = Input(shape=(G.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    shared_layers = nn_base(img_input, trainable=True)

    num_anchors = len(G.anchor_box_scales) * len(G.anchor_box_ratios)
    rpn_layers = rpn(shared_layers, num_anchors)
    classifier_layers = classifier(feature_map_input, roi_input, G.num_rois, nb_classes=len(class_mapping), trainable=True)

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
    G.filteredROIs = []
    print(G.class_mapping)
    X, ratio = IP.format_img(img)
    G.ratio = ratio
    X = np.transpose(X, (0, 2, 3, 1))
    with G.graph.as_default():
        #Proposes regions on the image X, for all anchor and points on the image, gives a sigmoid class and a regr value.
        #cls_sigmoid => (1,37,60,9): (37,60) is the kernel size of the last base CNN layer. For each point in the kernel, for all 9 anchor points, gets the sigmoid value (0 to 1)
        #bbox_regr => (1,37,60,9*4): 4 coord for each anchor point
        #base_layers => (1,37,60,512): 512 is no of output filters at the end of chosen CNN layers.
        [cls_sigmoid, bbox_regr, base_layers] = G.model_rpn.predict(X)

        #Debug
        (G.cls_sigmoid, G.bbox_regr, G.base_layers) = (cls_sigmoid, bbox_regr, base_layers)

        #Select the proper top 300 from all RPN values and get the ROI (x1,y1,x2,y2) based on overlap threshold.
        #R => (300, 4): 300 * (x1,y1,x2,y2)
        R = roi_helpers.rpn_to_roi(cls_sigmoid, bbox_regr, K.common.image_dim_ordering(), overlap_thresh=0.7, debug=False)
        G.ROIs = R
        print("Original ROIs ", G.ROIs.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

         #print(R.shape[0]) #300//32 => 9 pools with 32 ROI
        for jk in range(R.shape[0]//G.num_rois + 1):
            pyramid_ROIs = np.expand_dims(R[G.num_rois*jk:G.num_rois*(jk+1), :], axis=0)
            if pyramid_ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//G.num_rois:
                continue

            #For 32 ROIs, predict probability (all 21 classes) and regr boxes (matches the class?)
            #P_regr: 4 coord of Regression for each class (out of 20). 20 classes minus 'bg'
            [P_cls, P_regr] = G.model_classifier.predict([base_layers, pyramid_ROIs])

            #print(P_cls.shape) #(1,32,21) (1,num_rois, num_classes)
            #print(P_regr.shape) #(1,32,80) (1,num_rois,4*(num_classes-1))

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < G.bbox_threshold:
                    continue
                max_cls_idx = np.argmax(P_cls[0, ii, :])
                #print(G.class_mapping[max_cls_idx])
                cls_name = G.class_mapping[max_cls_idx]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = pyramid_ROIs[0, ii, :]

                #Every cls has correspongin 4 coords of reg. So we select reg of max cls only from P_regr
                try:
                    #Get Regr of only the selected Class using cls_num
                    (tx, ty, tw, th) = P_regr[0, ii, 4*max_cls_idx:4*(max_cls_idx+1)]
                    #classifier_regr_std gives the proper scale to apply regr.
                    tx /= G.classifier_regr_std[0]
                    ty /= G.classifier_regr_std[1]
                    tw /= G.classifier_regr_std[2]
                    th /= G.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                       
                ## Debug
                (x1, y1, x2, y2) = (x, y, w+x, h+y)
                G.tot_SPPs.append([x1, y1, x2, y2])

                bboxes[cls_name].append([G.rpn_stride*x, G.rpn_stride*y, G.rpn_stride*(x+w), G.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        
        #G.d_bbox = np.array(bboxes['person'])
        #G.d_prob = np.array(probs['person'])

        #print(len(bboxes))
        # for key in bboxes:
        #     bbox_arr = np.array(bboxes[key])
        #     prob_arr = np.array(probs[key])
        #     print(key, len(bbox_arr))
        #     new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox_arr,prob_arr,overlap_thresh=0.5)
        #     for jk in range(new_boxes.shape[0]):
        #         (x1, y1, x2, y2) = new_boxes[jk,:]
        #         if key != 'bg':
        #             G.filteredROIs.append([x1, y1, x2, y2])
        #             textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))

        G.tot_SPPs = np.array([G.tot_SPPs])
        G.tot_SPPs = np.squeeze(G.tot_SPPs, axis=0)
        print("filteredROIs ", G.tot_SPPs.shape)

################################ DEBUG VIZ Funcs ######################################################
#Get filtered RPNs based on input for UI Debug Viz
def d_resetFilteredRpns():
    #If debug=True, the algo returns max 300 RPNs just like original, but this time filtered by options
    R = roi_helpers.rpn_to_roi(G.cls_sigmoid, G.bbox_regr, K.common.image_dim_ordering(), overlap_thresh=0.7, debug=True)
    G.ROIs = R
    print("Filtered ROIs ", G.ROIs.shape)

def d_getSinglePyramid(pool_i):
    #If debug=True, the algo returns max 300 RPNs just like original, but this time filtered by options
    R = roi_helpers.rpn_to_roi(G.cls_sigmoid, G.bbox_regr, K.common.image_dim_ordering(), overlap_thresh=0.7, debug=False)
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    singlePool = []
    jk = pool_i

    pyramid_ROIs = np.expand_dims(R[G.num_rois*jk:G.num_rois*(jk+1), :], axis=0)
    if jk == R.shape[0]//G.num_rois:
        #pad R
        curr_shape = pyramid_ROIs.shape
        target_shape = (curr_shape[0],G.num_rois,curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(pyramid_ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = pyramid_ROIs
        ROIs_padded[0, curr_shape[1]:, :] = pyramid_ROIs[0, 0, :]
        pyramid_ROIs = ROIs_padded

    [P_cls, P_regr] = G.model_classifier.predict([G.base_layers, pyramid_ROIs])
    for ii in range(P_cls.shape[1]):
        if np.max(P_cls[0, ii, :]) < G.bbox_threshold:
            continue
        (x, y, w, h) = pyramid_ROIs[0, ii, :]
        ## Debug
        (x1, y1, x2, y2) = (x, y, w+x, h+y)
        singlePool.append([x1, y1, x2, y2])
    
    singlePool = np.array([singlePool])
    singlePool = np.squeeze(singlePool, axis=0)
    print(jk, "-, singlePool ", singlePool.shape)
    return singlePool

def d_getRpnToRoi(i_start=0, i_end=10, ratios=[], sizes=[]):
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
        print("re get rpns")
        d_resetFilteredRpns()
        Debug_R = G.ROIs[i_start:i_end,:]
    
    #Debug: Display chosen ROIs on the image.
    print("display box")
    fig = draw_plots.drawBoxes(G.debug_img, boxes=Debug_R, scale=G.rpn_stride)
    return fig

def d_getPyramidPools():
    Debug_P = G.tot_SPPs
    fig = draw_plots.drawBoxes(G.debug_img, boxes=Debug_P, scale=G.rpn_stride)
    return fig

def d_getSinglePool(pool_i):
    Debug_SP = d_getSinglePyramid(pool_i)
    fig = draw_plots.drawBoxes(G.debug_img, boxes=Debug_SP, scale=G.rpn_stride)
    return fig

def getNonmaxSuppression(nonMax_i):
    print("NonMax ", nonMax_i)
    # grab the coordinates of the bounding boxes
    x1 = G.d_bbox[:, 0]
    y1 = G.d_bbox[:, 1]
    x2 = G.d_bbox[:, 2]
    y2 = G.d_bbox[:, 3]
    # calculate the areas
    area = (x2 - x1) * (y2 - y1)
    if nonMax_i == 0:
        # initialize the list of picked indexes
        G.picked_idxs = []
        G.sorted_idxs = np.argsort(G.d_prob)

    last = len(G.sorted_idxs) - 1
    i = G.sorted_idxs[last]
    G.picked_idxs.append(i)
    G.sorted_idxs = np.delete(G.sorted_idxs, [last])
    print(i, G.sorted_idxs)
    print(x1[i], y1[i], x2[i], y2[i])
    fig = draw_plots.showOverlapBoxes(G.debug_img, selectedRect=[x1[i], y1[i], x2[i], y2[i]])
    return fig
    
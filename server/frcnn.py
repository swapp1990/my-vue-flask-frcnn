import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from RoiPoolingConv import RoiPoolingConv

anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
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

    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
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
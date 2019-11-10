#Common
global client_img
global active_threads
active_threads = []

global img_gen

#Constants
global class_mapping
class_mapping = {'tvmonitor': 0, 'train': 1, 'person': 2, 'boat': 3, 'horse': 4, 'cow': 5, 'bottle': 6, 'dog': 7, 'aeroplane': 8, 'car': 9, 'bus': 10, 'bicycle': 11, 'chair': 12, 'diningtable': 13, 'pottedplant': 14, 'bird': 15, 'cat': 16, 'motorbike': 17, 'sheep': 18, 'sofa': 19, 'bg': 20}

global anchor_box_scales, anchor_box_ratios
anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

global img_channel_mean
img_channel_mean = [103.939, 116.779, 123.68]

global im_size
im_size = 600

global img_scaling_factor
img_scaling_factor = 1.0

global ratio
ratio = 1.0

global classifier_regr_std
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

global rpn_stride, num_rois
rpn_stride = 16
num_rois = 32

#Objects
global model_rpn, model_classifier, graph
global ROIs

#Debug
global debug_img
global anchor_scales_d, anchor_ratios_d
global cls_sigmoid, bbox_regr, base_layers
global d_bbox, d_prob
global bbox_threshold
global picked_idxs, sorted_idxs
picked_idxs = []
#nonmax
#For given current nonmax step
global selectedBox, selectedTxt, allBoxes, allTexts, overlapBoxes, pickedBoxes
global singleNonOverlaps
singleNonOverlaps = []

anchor_scales_d = []
anchor_ratios_d = []
bbox_threshold = 0.7

global tot_SPPs
tot_SPPs = []
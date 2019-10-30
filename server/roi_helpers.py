import numpy as np
import pdb
import math
import copy
import time

import settings
import utils
from matplotlib import pyplot as plt
import global_vars as G

rpn_stride = 16
im_size = 600
# overlaps for classifier ROIs
classifier_min_overlap = 0.1
classifier_max_overlap = 0.5
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

# image resize
def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h

# Intersection of Union
def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

def calc_iou(R, img_data, class_mapping):

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if classifier_min_overlap <= best_iou < classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    picked_idxs = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)
    # sort the bounding boxes
    #Probs gives high probability for all bboxes including 'bg'. But we want a mixture of 'bg' and other classes.
    sorted_idxs = np.argsort(probs)
    picked_idxs = []
    
    stepRecord = 0
    # keep looping while some indexes still remain in the indexes
    # list
    while len(sorted_idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(sorted_idxs) - 1
        i = sorted_idxs[last]
        picked_idxs.append(i)

        #Debug Steps to capture
        # stepsToCapture = [1]#[1,5,50,150,290,300]
        # if len(picked_idxs) in stepsToCapture:
        #     print("stepRecord ", stepRecord)
        #     showOverlapBoxes(boxes, i, sorted_idxs)
        # find the intersection

        xx1_int = np.maximum(x1[i], x1[sorted_idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[sorted_idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[sorted_idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[sorted_idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        #find the union
        area_union = area[i] + area[sorted_idxs[:last]] - area_int

        #compute the ratio of overlap
        overlap_arr = area_int/(area_union + 1e-6)
        #Filter overlapping boxes and remove them including selected box (last) from sorted_idxs
        filter_idxs = np.where(overlap_arr > overlap_thresh)[0]

        #delete all indexes from the index list that have
        delete_idxs = np.concatenate(([last], filter_idxs))
        sorted_idxs = np.delete(sorted_idxs, delete_idxs)

        #filter_idxs = np.where(overlap_arr > 0.9)[0]
        #Debug Steps to capture
        # stepsToCapture = [1,5,50,150,290,300]
        # if len(picked_idxs) in stepsToCapture:
        #     print("stepRecord ", stepRecord)
        #     showPickedBoxes(boxes, sorted_idxs, picked_idxs, filter_idxs)
        #stepRecord += 1

        if len(picked_idxs) >= max_boxes:
            break
    
    
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[picked_idxs].astype("int")
    probs = probs[picked_idxs]
    return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9, debug=False):
    std_scaling = 4.0
    regr_layer = regr_layer / std_scaling

    if debug:
        anchor_sizes = G.anchor_scales_d
        anchor_ratios = G.anchor_ratios_d
    else:
        anchor_sizes = G.anchor_box_scales
        anchor_ratios = G.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0])/rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1])/rpn_stride
            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols),np. arange(rows))

            A[0, :, :, curr_layer] = X - anchor_x/2
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0))
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    print(len(all_boxes))
    #return top 300 bboxes
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    #showImage(result)

    return result

######################### DEBUG ####################################################
def showOverlapBoxes(boxes, currBox_idx, other_idxs):
    last = len(other_idxs) - 1
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    i = currBox_idx

    xx1_int = np.maximum(x1[i], x1[other_idxs[:last]])
    yy1_int = np.maximum(y1[i], y1[other_idxs[:last]])
    xx2_int = np.minimum(x2[i], x2[other_idxs[:last]])
    yy2_int = np.minimum(y2[i], y2[other_idxs[:last]])

    ww_int = np.maximum(0, xx2_int - xx1_int)
    hh_int = np.maximum(0, yy2_int - yy1_int)

    area_int = ww_int * hh_int

    #find the union
    area_union = area[i] + area[other_idxs[:last]] - area_int

    #compute the ratio of overlap
    overlap_arr = area_int/(area_union + 1e-6)
    #Filter overlapping boxes and remove them including selected box (last) from sorted_idxs
    filter_idxs = np.where(overlap_arr > 0.9)[0]
    print("overlap ", filter_idxs)
    # img_copy = settings.myDebugList['img'].copy()


    # i = currBox_idx
    # utils.addRectToImg(img_copy,x1[i]*rpn_stride, y1[i]*rpn_stride, x2[i]*rpn_stride, y2[i]*rpn_stride, scaled=True, color=(0,255,0), thickness=2)

    # last = len(other_idxs) - 1
    # xx1_int = np.maximum(x1[i], x1[other_idxs[:last]])
    # yy1_int = np.maximum(y1[i], y1[other_idxs[:last]])
    # xx2_int = np.minimum(x2[i], x2[other_idxs[:last]])
    # yy2_int = np.minimum(y2[i], y2[other_idxs[:last]])

    # maxRect = {"i":-1, "size":0, "rect":[]}
    # minRect = {"i":-1, "size":1000, "rect":[]}

    # currCount = 0
    # maxRange = len(other_idxs)-1
    # maxBoxes = maxRange
    # for j in range(maxRange):
    #     #j = j + 5000
    #     ww_int = np.maximum(0, xx2_int[j] - xx1_int[j])
    #     hh_int = np.maximum(0, yy2_int[j] - yy1_int[j])
    #     #Area of intersection with picked rect
    #     area_int = ww_int * hh_int

    #     #Area of union of picked rect and selected rect
    #     area_union = area[i] + area[j] - area_int

    #     #overlapping area
    #     overlap = area_int/(area_union + 1e-6)

    #     #if overlap > 0.1 and overlap < 0.2 and currCount != maxBoxes:
    #     if currCount != maxBoxes:
    #         currCount += 1
    #         utils.addRectToImg(img_copy,xx1_int[j]*rpn_stride, yy1_int[j]*rpn_stride, xx2_int[j]*rpn_stride, yy2_int[j]*rpn_stride, scaled=True, color=(255,255,0), thickness=1)
    #         maxRectSize = maxRect["size"]
    #         if maxRectSize < area[j]:
    #              maxRect["size"] = area[j]
    #              maxRect["i"] = j
    #              maxRect["rect"]= [xx1_int[j]*rpn_stride, yy1_int[j]*rpn_stride, xx2_int[j]*rpn_stride, yy2_int[j]*rpn_stride]
    #         minRectSize = minRect["size"]
    #         if minRectSize > area[j]:
    #             minRect["size"] = area[j]
    #             minRect["i"] = j
    #             minRect["rect"]= [xx1_int[j]*rpn_stride, yy1_int[j]*rpn_stride, xx2_int[j]*rpn_stride, yy2_int[j]*rpn_stride]
    #         #print(area[i], area[j], area_int, area_union, overlap)
        
    #     if currCount == maxBoxes:
    #         break
    # print("MaxRect ", maxRect["size"], " MinRect", minRect["size"])
    # #Display Min and Max Rect as seperate color
    # #Min Box
    # minBox = minRect["rect"]
    # utils.addRectToImg(img_copy,minBox[0], minBox[1], minBox[2], minBox[3], scaled=True, color=(0,255,255), thickness=2)
    # maxBox = maxRect["rect"]
    # utils.addRectToImg(img_copy,maxBox[0], maxBox[1], maxBox[2], maxBox[3], scaled=True, color=(0,255,255), thickness=2)
    # #print(len(xx1_int), len(yy1_int), len(xx2_int), len(yy2_int))

    # plt.imshow(img_copy)
    # plt.show()

def showPickedBoxes(boxes, left_idxs, picked_idxs, filtered_idxs):
    img_copy = settings.myDebugList['img'].copy()
    rpn_stride = settings.myDebugList['rpn_stride']

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    (x1, y1, x2, y2) = (x1*rpn_stride, y1*rpn_stride, x2*rpn_stride, y2*rpn_stride)

    print("Left ", len(left_idxs), " Removed", len(filtered_idxs))
    # for idx in range(len(left_idxs)):
    #     i = left_idxs[idx]
    #     utils.addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(255,0,0))
    if len(picked_idxs)>2:
        picked_idxs = picked_idxs[-2:]
    for idx in range(len(picked_idxs)):
        i = picked_idxs[idx]
        utils.addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(0,255,0), thickness=2)

    for idx in range(len(filtered_idxs)):
        i = filtered_idxs[idx]
        utils.addRectToImg(img_copy,x1[i], y1[i], x2[i], y2[i], scaled=True, color=(255,0,0))
    plt.imshow(img_copy)
    plt.show()

def showImage(boxes):
    colors=[(0,155,0), (255,0,0)]
    selColor = colors[0]
    img_copy = settings.myDebugList['img']
    ratio = settings.myDebugList['ratio']
    rpn_stride = settings.myDebugList['rpn_stride']
    for j in range(len(boxes)):
        (x1, y1, x2, y2) = boxes[j]
        (x1, y1, x2, y2) = (x1*rpn_stride, y1*rpn_stride, x2*rpn_stride, y2*rpn_stride)
        (x1, y1, x2, y2) = utils.get_real_coordinates(ratio, x1, y1, x2, y2)
        utils.addRectToImg(img_copy, x1, y1, x2, y2, color=selColor)
    plt.imshow(img_copy)
    plt.show()
    #cv2.waitKey(0)
    #cv2.imwrite(filename,img_copy)
    #print('Image saved ', filename)




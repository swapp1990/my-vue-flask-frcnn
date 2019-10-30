export const rpnTpRoi_code = `
def rpn_to_roi(rpn_layer, regr_layer, dim_ordering, use_regr=True, max_boxes=300,overlap_thresh=0.9, debug=False):
    regr_layer = regr_layer / std_scaling

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

    #return top 300 bboxes
    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result`;

export const SPP_code = `
for jk in range(R.shape[0]//num_rois + 1):
  pyramid_ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1), :], axis=0)
  if pyramid_ROIs.shape[1] == 0:
      break

  if jk == R.shape[0]//num_rois:
      continue

  [P_cls, P_regr] = model_classifier.predict([base_layers, pyramid_ROIs])
  for ii in range(P_cls.shape[1]):
    if np.max(P_cls[0, ii, :]) < G.bbox_threshold:
      continue
    max_cls_idx = np.argmax(P_cls[0, ii, :])
    (x, y, w, h) = pyramid_ROIs[0, ii, :]

    (tx, ty, tw, th) = P_regr[0, ii, 4*max_cls_idx:4*(max_cls_idx+1)]
    tx /= G.classifier_regr_std[0]
    ty /= G.classifier_regr_std[1]
    tw /= G.classifier_regr_std[2]
    th /= G.classifier_regr_std[3]
    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

    (x1, y1, x2, y2) = (x, y, w+x, h+y)
    `

export const NMS_code = `
result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

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
    # Probs gives high probability for all bboxes including 'bg'. But we want a mixture of 'bg' and other classes.
    sorted_idxs = np.argsort(probs)
    picked_idxs = []

    # keep looping while some indexes still remain in the indexes
    # list
    while len(sorted_idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(sorted_idxs) - 1
        i = sorted_idxs[last]
        picked_idxs.append(i)

        xx1_int = np.maximum(x1[i], x1[sorted_idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[sorted_idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[sorted_idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[sorted_idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int
        area_union = area[i] + area[sorted_idxs[:last]] - area_int
        overlap_arr = area_int/(area_union + 1e-6)

        #Filter overlapping boxes and remove them including selected box (last) from sorted_idxs
        filter_idxs = np.where(overlap_arr > overlap_thresh)[0]

        #delete all indexes from the index list that have
        delete_idxs = np.concatenate(([last], filter_idxs))
        sorted_idxs = np.delete(sorted_idxs, delete_idxs)

        filter_idxs = np.where(overlap_arr > 0.9)[0]

        if len(picked_idxs) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[picked_idxs].astype("int")
    probs = probs[picked_idxs]
    return boxes, probs
    `
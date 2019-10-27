import random
import cv2
from threading import Lock
from matplotlib import pyplot as plt
import mpld3
from mpld3 import plugins

import settings
import global_vars as g

lock = Lock()

############################ Images #############################################
def draw_img(img):
    img_copy = img.copy()
    #addRectToImg(img_copy, 20,30,70,90, scaled=True)
    fig, ax = plt.subplots()
    ax.imshow(img_copy)
    return mpld3.fig_to_html(fig)

def displayBoxes(img, anchors=[], rois=[], frois=[], showBox=True, texts=[]):
    img_copy = img.copy()
    colorbb = (0, 255, 0)
    coloraa = (255, 0, 0)
    
    for i in range(len(anchors)):
        a = anchors[i]
        txt = texts[i]
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        cv2.putText(img_copy, txt, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA) 
        if showBox:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), coloraa, 2)
        cv2.circle(img_copy, (int((x1+x2)/2), int((y1+y2)/2)), 3, coloraa, -1)

    for i in range(len(rois)):
        roi = rois[i]
        x1, y1, x2, y2 = int(roi[0]*g.rpn_stride), int(roi[1]*g.rpn_stride), int(roi[2]*g.rpn_stride), int(roi[3]*g.rpn_stride)
        addRectToImg(img_copy, x1, y1, x2, y2, scaled=True, color=(0,255,0))
    for i in range(len(frois)):
        froi = frois[i]
        x1, y1, x2, y2 = int(froi[0]), int(froi[1]), int(froi[2]), int(froi[3])
        addRectToImg(img_copy, x1, y1, x2, y2, scaled=True, color=(0,0,255))
    #Debug
    #cv2.imwrite('result/roi_debug.png',img_copy)

    fig, ax = plt.subplots()
    ax.imshow(img_copy)
    return mpld3.fig_to_html(fig)
    
    # plt.figure(figsize=(8,8))
    # plt.imshow(img)
    # plt.show()

def showOverlapBoxes(img, selectedRect):
    img_copy = img.copy()
    x1, y1, x2, y2 = int(selectedRect[0]), int(selectedRect[1]), int(selectedRect[2]), int(selectedRect[3])
    addRectToImg(img_copy, x1, y1, x2, y2, scaled=True)
    fig, ax = plt.subplots()
    ax.imshow(img_copy)
    return mpld3.fig_to_html(fig)

def addRectToImg(img, x1, y1, x2, y2, scaled=False, color=(255,0,0), thickness=1):
    if scaled:
        ratio = g.ratio
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

    #coloraa = (255, 0, 0)
    cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, color, 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)
############################ Graphs #############################################
x = range(100)
y = [a * 2 + random.randint(-20, 20) for a in x]
pie_fracs = [20, 30, 40, 10]
pie_labels = ["A", "B", "C", "D"]
def draw_fig(fig_type):
    """Returns html equivalent of matplotlib figure
    Parameters
    ----------
    fig_type: string, type of figure
            one of following:
                    * line
                    * bar
    Returns
    --------
    d3 representation of figure
    """

    with lock:
        fig, ax = plt.subplots()
        if fig_type == "line":
            ax.plot(x, y)
        elif fig_type == "bar":
            ax.bar(x, y)
        elif fig_type == "pie":
            ax.pie(pie_fracs, labels=pie_labels)
        elif fig_type == "scatter":
            ax.scatter(x, y)
        elif fig_type == "hist":
            ax.hist(y, 10, normed=1)
        elif fig_type == "area":
            ax.plot(x, y)
            ax.fill_between(x, 0, y, alpha=0.2)

    return mpld3.fig_to_html(fig)


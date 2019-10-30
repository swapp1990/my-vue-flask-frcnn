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

#scale: Scales coordinates on image if box coord are downscaled by a factor of int value
def drawBoxes(img, boxes=[], scale=1, showBox=True, texts=[]):
    img_copy = img.copy()

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = int(box[0]*scale), int(box[1]*scale), int(box[2]*scale), int(box[3]*scale)
        addRectToImg(img_copy, x1, y1, x2, y2, scaled=True, color=(0,255,0))

    #Debug
    # cv2.imwrite('result/roi_debug.png',img_copy)
    # plt.figure(figsize=(8,8))
    # plt.imshow(img)
    # plt.show()

    fig, ax = plt.subplots()
    ax.imshow(img_copy)
    return mpld3.fig_to_html(fig)

def showOverlapBoxes(img, selectedRect=None, selectedTxt="", overlapBoxes=[], overlapTexts=[], color=(255,255,0)):
    img_copy = img.copy()
    if selectedRect != None:
        x1, y1, x2, y2 = int(selectedRect[0]), int(selectedRect[1]), int(selectedRect[2]), int(selectedRect[3])
        addRectToImg(img_copy, x1, y1, x2, y2, scaled=True, txt=selectedTxt, thickness=2)
    for i in range(len(overlapBoxes)):
        box = overlapBoxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        txt = ""
        if len(overlapTexts) > 0:
            txt = overlapTexts[i]
        addRectToImg(img_copy, x1, y1, x2, y2, scaled=True, color=color, txt=txt)

    fig, ax = plt.subplots()
    ax.imshow(img_copy)
    return mpld3.fig_to_html(fig)

def addRectToImg(img, x1, y1, x2, y2, scaled=False, color=(255,0,0), thickness=1, txt=""):
    if scaled:
        ratio = g.ratio
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

    #coloraa = (255, 0, 0)
    if txt!="":
        cv2.putText(img, txt, (x1, y1+(y2-y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
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


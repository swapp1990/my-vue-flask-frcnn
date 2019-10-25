import cv2
import settings

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)

def addRectToImg(img, x1, y1, x2, y2, scaled=False, color=(255,0,0), thickness=1):
    if scaled:
        ratio = settings.myDebugList['ratio']
        (x1, y1, x2, y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

    #coloraa = (255, 0, 0)
    cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 3, color, 1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
import numpy as np
import cv2
import math
try:
    from cpp_bindings.cpp_bindings import find_char_boxes, find_word_boxes
    CPP_BIND_AVAILABLE = True
except BaseException as e:
    CPP_BIND_AVAILABLE = False

""" auxilary functions """
# unwarp corodinates
@@ -15,8 +20,36 @@ def warpCoord(Minv, pt):
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """

def getCharBoxes(image, textmap, use_cpp_bindings=True):
    char_boxes = []
    ret, sure_fg = cv2.threshold(textmap, 0.6, 1, 0)
    ret, sure_bg = cv2.threshold(textmap, 0.2, 1, 0)

    sure_fg = np.uint8(sure_fg * 255)
    sure_bg = np.uint8(sure_bg * 255)

    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    image = cv2.resize(image, textmap.shape[::-1], cv2.INTER_CUBIC)
    cv2.watershed((image * 255).astype(np.uint8), markers)
    num_classes = np.max(markers)

    # marker 1 is background
    if CPP_BIND_AVAILABLE and use_cpp_bindings:
        char_boxes = find_char_boxes(markers, num_classes)
    else:
        for i in range(2, np.max(markers) + 1):
            np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
            l, t, w, h = cv2.boundingRect(np_contours)
            box = np.array([[l, t], [l + w, t], [l + w, t + h], [l, t + h]], dtype=np.float32)
            char_boxes.append(box)

    return char_boxes

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text,
        use_cpp_bindings=True, fast_mode=True, rotated_box=True):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
@@ -29,52 +62,56 @@ def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
 if CPP_BIND_AVAILABLE and use_cpp_bindings:
        det, mapper = find_word_boxes(textmap, labels, nLabels, stats,
                text_threshold, fast_mode, rotated_box)
    else:
        det = []
        mapper = []
        for k in range(1,nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10: continue

            # thresholding
            if np.max(textmap[labels==k]) < text_threshold: continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels==k] = 255
            segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0 : sx = 0
            if sy < 0 : sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:,0]), max(np_contours[:,0])
                t, b = min(np_contours[:,1]), max(np_contours[:,1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4-startidx, 0)
            box = np.array(box)

            det.append(box)
            mapper.append(k)

    return det, labels, mapper

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text,
        poly=False, use_cpp_bindings=True, fast_mode=False, rotated_box=True):

    boxes, labels, mapper = getDetBoxes_core(
            textmap, linkmap, text_threshold, link_threshold,
            low_text, use_cpp_bindings, fast_mode, rotated_box)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
@@ -234,6 +275,18 @@ def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly

    return boxes, polys

def getWordAndCharBoxes(image, textmap, linkmap, text_threshold, link_threshold,
        low_text, poly=False, use_cpp_bindings=True, fast_mode=False, rotated_box=True):

    boxes, polys = getDetBoxes(
            textmap, linkmap, text_threshold, link_threshold,
            low_text, poly, use_cpp_bindings, fast_mode, rotated_box)

    char_boxes = getCharBoxes(image, textmap)

    return boxes, polys, char_boxes


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
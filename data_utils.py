import os, sys, re, six, math, lmdb, torch, cv2
import numpy as np
from natsort import natsorted
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torchvision.transforms as T
import torch.nn.functional as F
from functools import reduce
import operator

def load_character_list(path) :
    with open(path, 'r') as f :
        character_list = f.read()
        
    return character_list

def fit_line(p1, p2):
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle



def rectangle_from_parallelogram(poly):
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        if angle/np.pi * 180 > 45:
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # 这个点为p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle
        
        
def sort_rectangle_custom(poly, debug = False) :
    
    idx_ls = list(range(len(poly)))
    
    func = lambda x : np.linalg.norm(x[1])
    p0_index = min(((i, coord) for i, coord in enumerate(poly)), key = func)[0]
    p0 = poly[p0_index]
    idx_ls.pop(p0_index)
    if debug :
        print(poly)
        
        print('left top point : ', p0_index)
        print('indices : ', idx_ls)
    
    def cal_angle(pt1, pt2, vertical = True) :
        
        if vertical :
            if pt2[1] - pt1[1] == 0 :
                return np.inf
            return np.arctan(abs(pt1[0] - pt2[0])/abs(pt2[1] - pt1[1]))
        else :
            if pt2[0] - pt1[0] == 0 :
                return np.inf
            return np.arctan(abs(pt1[1]-pt2[1])/abs(pt2[0] - pt1[0]))
    
    vertical_angles = [(cal_angle(poly[p0_index], poly[idx], vertical = True), idx) for idx in idx_ls]
    if debug :
        print(vertical_angles)
    horizontal_angles = [(cal_angle(poly[p0_index], poly[idx], vertical = False), idx) for idx in idx_ls]
    if debug :
        print(horizontal_angles)
    
    p1_index = min(horizontal_angles, key = lambda x : x[0])[1]
    p3_index = min(vertical_angles, key = lambda x : x[0])[1]
    p1 = poly[p1_index]
    p3 = poly[p3_index]
    
    idx_ls.remove(p1_index)
    idx_ls.remove(p3_index)
    p2 = poly[idx_ls[0]]
    
    return np.array([p0, p1, p2, p3])
    

def is_convex_polygon(polygon):
    TWO_PI = 2 * np.pi
    try:  
        if len(polygon) < 3:
            return False
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        for ndx, newpoint in enumerate(polygon):
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += TWO_PI 
            elif angle > np.pi:
                angle -= TWO_PI
            if ndx == 0:  
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  
                if orientation * angle <= 0.0:  
                    return False
            angle_sum += angle
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  
    
    
def sort_rectangle_centroid(coords) :
    
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x,y), coords), [len(coords)]*2))
    recoord = sorted(coords, key = lambda coord: (180+math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1])))*360)
    
    return np.array(recoord)


def create_orientation(affinity_bboxes, character_bboxes, region_scores, words) :
    '''takes affinitypoints(straight links) and character bbox coordinates
       return orientation_x and orientation_y with image size'''
    
    orientation_x = Image.new('L',region_scores.shape[::-1][:2], 0 )
    orientation_x_draw = ImageDraw.Draw(orientation_x)
    orientation_y = Image.new('L', region_scores.shape[::-1][:2], 0)
    orientation_y_draw = ImageDraw.Draw(orientation_y)
    
    scale_region = region_scores.copy()
    scale_region[scale_region<255*0.5] = 0
    scale_region[scale_region>255*0.5] = 1

#     assert len(affinity_bboxes) == len(character_bboxes)
    for affpoints, character_bbox, word in zip(affinity_bboxes, character_bboxes, words) :
#         assert len(affpoints) == len(character_bbox)
        
        for i, charboxes in enumerate(character_bbox) :
            
            if len(character_bbox) == 1 or len(affpoints) == 0 : 
                #TODO : [ROT{각도}] 태그 반영하기
                theta = 0
               
            else :
                
                if i != len(character_bbox)-1:
                    try:
                        x0,y0 = affpoints[i]
                        x1,y1 = affpoints[i+1]
                    except :
                        print(len(charboxes), len(affpoints))
                        raise ValueError
                    theta = np.arctan2((y1-y0),(x1-x0)) #TODO : 글자 마지막에 대한 orientation 처리 필요
                    
            #         cv2.fillConvexPoly(orientation_x, charboxes, ori_x)
            #         cv2.fillConvexPoly(orientation_y, charboxes, ori_y)
                    
#                 else :
#                     theta = 0
                    

            #         cv2.fillConvexPoly(orientation_x, charboxes, ori_x)
            #         cv2.fillConvexPoly(orientation_y, charboxes, ori_y)
#             if word.startswith('[ROT') :
#                 theta = 1
            
            
            ori_x = (np.cos(theta*np.pi/2)+1)/2*255.
            ori_y = (np.sin(theta*np.pi/2)+1)/2*255.
            orientation_x_draw.polygon(charboxes.flatten().tolist(),fill = int(ori_x))
            orientation_y_draw.polygon(charboxes.flatten().tolist(),fill = int(ori_y))

    orientation_x_region = np.multiply(scale_region, np.array(orientation_x)/255.)
    orientation_y_region = np.multiply(scale_region, np.array(orientation_y)/255.)
        
    return orientation_x_region, orientation_y_region #np.array(orientation_x)/255., np.array(orientation_y)/255.


def crop_image_by_bbox(image, box, word):

    rot_angle = None
    match = re.findall(r'\[ROT[0-9]+\]', word)
    if len(match) > 0  :
        rot_angle = int(re.findall(r'\d+', match[0])[0])
    word = word.replace('[UNK]', '*')
    word = re.sub(r'[UNK[0-9]+]|[ROT[0-9]+]', '', word)
    if len(box) == 4 :
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5 and len(word) != 1 :
            width = h
            height = w
            # [ROT90]일 때
            if rot_angle == 90 or rot_angle == None or rot_angle == 0  :
                M = cv2.getPerspectiveTransform(np.float32(box),
                                np.float32(np.array([[0, height], [0,0], [width, 0], [width, height]])))
            elif rot_angle == 270 :
                    M = cv2.getPerspectiveTransform(np.float32(box),
                                                np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
            else :
                M = cv2.getPerspectiveTransform(np.float32(box),
                                np.float32(np.array([[0, height], [0,0], [width, 0], [width, height]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M
    else : 
        # polygon(>4)
        pts = np.int32(box)
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        x,y = max(0, x), max(0, y)
        x,y = min(x, image.shape[1]), min(y, image.shape[0])
        cropped = np.array(image)[y:y+h, x:x+w]
        pts = pts - pts.min(axis = 0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255,255,255), -1, cv2.LINE_AA)
        cropped_black_bg = cv2.bitwise_and(cropped, cropped, mask=mask)
        bg = np.ones_like(cropped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask = mask)
        cropped_white_bg = bg + cropped_black_bg

        return cropped_white_bg, (x,y)


""" auxilary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

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
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)
        #kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        #segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel1, iterations=1)


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

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=True):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys



def scale_orientation(orientation_x, orientation_y, region_scores) :
    orientation_region = np.arctan(np.array(orientation_y) / np.array(orientation_x) )*255.
    mask = np.zeros_like(orientation_x)
    
    mask[np.isnan(orientation_region)] = 0
    mask[~np.isnan(orientation_region)] = 255
    return np.uint8(orientation_region) , np.uint8(mask)
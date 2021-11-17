import numpy as np
import math, cv2
from functools import reduce
import operator
from PIL import Image, ImageDraw
import torch

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
            
            ori_x = (np.cos(theta*np.pi/2)+1)/2*255.
            ori_y = (np.sin(theta*np.pi/2)+1)/2*255.
            orientation_x_draw.polygon(charboxes.flatten().tolist(),fill = int(ori_x))
            orientation_y_draw.polygon(charboxes.flatten().tolist(),fill = int(ori_y))

    orientation_x_region = np.multiply(scale_region, np.array(orientation_x)/255.)
    orientation_y_region = np.multiply(scale_region, np.array(orientation_y)/255.)
        
    return orientation_x_region, orientation_y_region
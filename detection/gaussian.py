import math
from math import exp
import numpy as np
import cv2
import os
import detection.imgproc as imgproc
from data_utils import sort_rectangle, sort_rectangle_custom, is_convex_polygon

def get_perpendicular_points(pt1, pt2, thickness) :
    '''get perpendicular points to line(pt1, pt2) with thickness, from pt1 and pt2 => total 4 points(pt1_top, pt2_top, pt2_bottom, pt1_bottom)'''

    x1,y1 = pt1
    x2,y2 = pt2

    slope = (y2 - y1)/ (x2 - x1)

    dy = math.sqrt(thickness**2/(slope**2+1))
    dx = -slope*dy

    tl = [x1+dx, y1+dy]
    tr = [x2+dx, y2+dy]
    br = [x2-dx, y2-dy]
    bl = [x2-dx, y2-dx]

    return np.array([tl, tr, br, bl])
    
    
class GaussianTransformer(object):

    def __init__(self, imgSize=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        distanceRatio = 3.34
        scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        self.region_threshold = region_threshold
        self.imgSize = imgSize
        self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize, distanceRatio)

        _, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        _, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def _gen_gaussian_heatmap(self, imgSize, distanceRatio):
        scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))
        heat = np.zeros((imgSize, imgSize), np.uint8)
        for i in range(imgSize):
            for j in range(imgSize):
                distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
                distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                heat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
        return heat

    def _test(self):
        sigma = 10
        spread = 3
        extent = int(spread * sigma)
        center = spread * sigma / 2
        gaussian_heatmap = np.zeros([extent, extent], dtype=np.float32)

        for i_ in range(extent):
            for j_ in range(extent):
                gaussian_heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                    -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))

        gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'test_guassian.jpg'), threshhold_guassian)

    def add_region_character(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
        return image

    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)
        
        affinity, _ = sort_rectangle(np.array([tl, tr, br, bl]))
        if is_convex_polygon(affinity) :
            return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)
        else :
            return image, None

        
    def add_adjust_affinity(self, image, bbox_1, bbox_2) :
        
        
        alpha = 0.5
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        
        x1_tl, y1_tl = bbox_1[0,:]
        x1_tr, y1_tr = bbox_1[1,:]
        x1_br, y1_br = bbox_1[2,:]
        x1_bl, y1_bl = bbox_1[3,:]
        
        x2_tl, y2_tl = bbox_2[0,:]
        x2_tr, y2_tr = bbox_2[1,:]
        x2_br, y2_br = bbox_2[2,:]
        x2_bl, y2_bl = bbox_2[3,:]
        
        d1 = min(math.sqrt((x1_br - x1_tl)**2 + (y1_br - y1_tl)**2), math.sqrt((x1_tr - x1_bl)**2 + (y1_tr - y1_bl)**2))
        d2 = min(math.sqrt((x2_br - x2_tl)**2 + (y2_br - y2_tl)**2), math.sqrt((x2_tr - x2_bl)**2 + (y2_tr - y2_bl)**2))
        thickness = int(max((d1+d2)/2*alpha, 1))
        
        pts = [center_1, center_2]
#         pts = pts.reshape((-1,1,2))
#         cv2.polylines(blank, [pts], False,(255), thickness)
        
#         dist = cv2.distanceTransform(blank, cv2.DIST_L2, 5)
#         dist = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
#         image[dist != 0] += dist[ dist!= 0 ] 
        
#         image = np.clip(image, 0,255)
        
        
#         affinity_line = sort_rectangle_custom(get_perpendicular_points(center_1, center_2, thickness))
        
        return pts, thickness
        
#         if is_convex_polygon(affinity_line) :
#             return self.add_affinity_character(image, affinity_line.copy()), np.expand_dims(affinity_line, axis=0)
#         else :
#             return image, None
        
    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                target = self.add_region_character(target, character_bbox[j])

        return target

    def generate_affinity(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                if target is not None and affinity is not None :
                    affinities.append(affinity)
                    total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities
    
    def generate_adjust_affinity(self, image_size, bboxes, words) :
        # TODO : returns polygon
        if len(bboxes) != len(words) :
            raise ValueError('# bboxes : {}, words : {} '.format(len(bboxes), words))
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            affinity_pts = []

            for char_num in range(character_bbox.shape[0] - 1):
                pts, thickness = self.add_adjust_affinity(target, character_bbox[total_letters], # TODO : thickness 글자간 변화 
                                                     character_bbox[total_letters + 1])
                if char_num == 0 :
                    affinity_pts.append(pts[0])
                    affinity_pts.append(pts[1])
                else :
                    affinity_pts.append(pts[1])
#                 if target is not None and affinity is not None :
#                     affinities.append(affinity)
                total_letters += 1
#         if len(affinities) > 0:
#             affinities = np.concatenate(affinities, axis=0)
            
            affinities.append(affinity_pts)
            blank = np.zeros([height, width], dtype=np.uint8)
            affinity_pts = np.array(affinity_pts, np.int32).reshape((-1,1,2))
            
            
            if character_bbox.shape[0] == 1 :
                thickness = 0 
            
#             try :
            cv2.polylines(blank, [affinity_pts], False,(255), thickness)
#             except :]

#                 thickness = 2
#                 cv2.polylines(blank, [affinity_pts], False,(255), thickness)

            dist = cv2.distanceTransform(blank, cv2.DIST_L2, 5)
            dist = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

            target[dist != 0] += dist[ dist!= 0 ] 

            target = np.clip(target, 0,255)
#         if len(affinities) > 0 :
#             affinities = np.concatenate(affinities, axis = 0)
    
        
    
        return target, affinities
    

    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.polylines(warped_color, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        threshhold = self.region_threshold * 255
        standardGaussianHeat1[standardGaussianHeat1 > 0] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.polylines(threshhold_guassian, [np.reshape(self.regionbox, (-1, 1, 2))], True, (255, 255, 255), thickness=1)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)

    
        

if __name__ == '__main__':
    gaussian = GaussianTransformer(512, 0.4, 0.2)
    gaussian.saveGaussianHeat()
    gaussian._test()
    bbox0 = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]])
    image = np.zeros((500, 500), np.uint8)
    # image = gaussian.add_region_character(image, bbox)
    bbox1 = np.array([[[100, 0], [200, 0], [200, 100], [100, 100]]])
    bbox2 = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]])
    bbox3 = np.array([[[0, 100], [100, 100], [100, 200], [0, 200]]])

    bbox4 = np.array([[[96, 0], [151, 9], [139, 64], [83, 58]]])
    image = gaussian.generate_region((500, 500, 1), [bbox4])
    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(image.copy() / 255)
    cv2.imshow("test", target_gaussian_heatmap_color)
    cv2.imwrite("test.jpg", target_gaussian_heatmap_color)
    cv2.waitKey()

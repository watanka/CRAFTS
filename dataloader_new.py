from config import config

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


import time, string
import re, random, itertools
from PIL import Image, ImageDraw
import scipy.io as scio
import Polygon as plg
from glob import glob
import matplotlib.pyplot as plt

from detection.gaussian import GaussianTransformer
from detection.watershed import watershed, watershed_wo_net
from detection.mep import mep
import detection.imgproc as imgproc
import detection.textsnake as textsnake
import detection.craft_utils as craft_utils

from recognition.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, TokenLabelConverter, Averager

from torchutil import copyStateDict
from file_utils import *
from data_utils import sort_rectangle_custom, create_orientation, load_character_list, is_convex_polygon


def ratio_area(h, w, box):
    area = h * w
    ratio = 0
    for i in range(len(box)):
        poly = plg.Polygon(box[i])
        box_area = poly.area()
        tem = box_area / area
        if tem > ratio:
            ratio = tem
    return ratio, area

def rescale_img(img, box, h, w):
    image = np.zeros((768,768,3),dtype = np.uint8)
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img
    box *= scale
    return image

def random_scale(img, bboxes, min_size):
    h, w = img.shape[0:2]
    # ratio, _ = ratio_area(h, w, bboxes)
    # if ratio > 0.5:
    #     image = rescale_img(img.copy(), bboxes, h, w)
    #     return image
    scale = 1.0
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
    random_scale = np.array([0.5, 1.0, 1.5, 2.0])
    scale1 = np.random.choice(random_scale)
    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1
    bboxes *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def padding_image(image,imgsize):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w <= tw or h <= th:
        return [padding_image(img, tw) for img in imgs]

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    #### IC15 for 0.6, MLT for 0.35 #####
    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


class Dataset(data.Dataset) :
    def __init__(self, 
                 cfg, 
                 delimiter = ',',  
                 orientation = True, 
                 target_size = 768, 
                 viz = False, 
                 debug = False,  
                 watershed_on = False) :
        self.viz = viz
        self.cfg = cfg
        self.std_cfg = config.Config(self.cfg.STD_CONFIG_PATH)
        self.str_cfg = config.Config(self.cfg.STR_CONFIG_PATH)
        self.data_folder = self.cfg.DATA_PATH
        
        # from STR configuration
        self.str_batch_size = self.str_cfg.str_batch_size
        self.PAD = self.str_cfg.PAD
        
        self.target_size = target_size
        self.delimiter = delimiter
        self.debug = debug
        self.watershed_on = watershed_on
        self.orientation = orientation
        
        self.images_path = []
        if type(self.data_folder) == list :
            
            for folder in self.data_folder :
                imgfiles = glob(os.path.join(folder, 'Train_images', '*.jpg')) + glob(os.path.join(folder, 'Train_images', '*.png'))
                gts = [imgfile.replace('Train_images','Train_gts').replace('.jpg','.txt').replace('.png','.txt') for imgfile in imgfiles]
                self.images_path.extend(imgfiles)
        else :
            imgfiles = glob(os.path.join(self.data_folder, 'Train_images', '*.jpg')) + glob(os.path.join(self.data_folder, 'Train_images', '*.png'))
            for imgfile in imgfiles :
                self.images_path.append(imgfile)
        self.gaussianTransformer = GaussianTransformer(imgSize=1024, region_threshold=0.5, affinity_threshold=0.15)
        

        character_list = load_character_list(self.str_cfg.CHAR_PATH)
        
        if self.str_cfg.ViTSTR:
            self.converter = TokenLabelConverter(character_list, self.str_cfg)
        else:
            self.converter = AttnLabelConverter(character_list)
        
    def __getitem__(self, index):
        return self.pull_item(index)
    
    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]
    
    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))
    
    def resize_channel_Gt(self, gtmask) :
        return cv2.cvtColor(cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2)), cv2.COLOR_RGB2GRAY)
    
    
    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8-sig').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(self.delimiter)     
            if len(ori_box) == 1:
                if self.delimiter == ',' :
                    delimiter = '\t'
                else :
                    delimiter = ','
                ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(delimiter)
            
            if len(ori_box) == 9 :
                box = [int(float(ori_box[j])) for j in range(8)]
                word = ori_box[-1].replace('@@@','')
                if word == '###' or word == '' :
                    word = '[UNK]'
                word = word.replace('###', '')
                box = np.array(box, np.int32).reshape(4, 2)
#                 box = sort_rectangle_custom(box)
                match = re.findall(r'\[UNK[0-9]+\]', word)

                if len(match) > 0 :
                    num = max(int(re.findall(r'\d+', match[0])[0]),1)
                    word = '[UNK]'*num

            else :
                if len(ori_box)<9 :
                    continue
                box = [int(float(ori_box[j])) for j in range(len(ori_box)-1)]
                word = ori_box[-1].replace('@@@','')
                if word == '###' or word == '' :
                    word = '[UNK]'
                word = word.replace('###', '')
                match = re.findall(r'\[UNK[0-9]+\]', word)

                if len(match) >0 :
                    #TODO : [ROT]에 대해서 처리
                    num = max(int(re.findall(r'\d+', match[0])[0]),1)
                    word = '[UNK]'*num
                box = np.array(box, np.int32).reshape(-1,2)
            bboxes.append(box)
            words.append(word)
        if len(bboxes) != len(words) :
            raise ValueError('# bboxes : {}, words : {} '.format(len(bboxes), words))

        return bboxes, words

    def load_gt(self, gt_path) : 
        lines = open(gt_path, encoding='utf-8-sig').readlines()
        polygons = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(self.delimiter)     
            if len(ori_box) == 1:
                if self.delimiter == ',' :
                    delimiter = '\t'
                else :
                    delimiter = ','
                ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(delimiter)
            
            if len(ori_box) == 9 :
                box = [int(float(ori_box[j])) for j in range(8)]
                word = ori_box[-1].replace('@@@','')
                if word == '###' or word == '' :
                    word = '[UNK]'
                word = word.replace('###', '')
                box = np.array(box, np.int32).reshape(4, 2)
                match = re.findall(r'\[UNK[0-9]+\]', word)

                if len(match) > 0 :
                    num = max(int(re.findall(r'\d+', match[0])[0]),1)
                    word = '[UNK]'*num

            else :
                if len(ori_box)<9 :
                    continue
                box = np.array([int(float(ori_box[j])) for j in range(len(ori_box)-1)], np.int32).reshape(-1,2)
                word = ori_box[-1].replace('@@@','')
                if word == '###' or word == '' :
                    word = '[UNK]'
                word = word.replace('###', '')
                match = re.findall(r'\[UNK[0-9]+\]', word)

                if len(match) >0 :
                    #TODO : [ROT]에 대해서 처리
                    num = max(int(re.findall(r'\d+', match[0])[0]),1)
                    word = '[UNK]'*num
            
            polygon = textsnake.TextInstance(box, 'c', word)
            if not is_convex_polygon(box) :
                polygon.checked = True
                if len(box) == 4 :
                    n_disk = 1
                else :
                    n_disk = len(polygon.text)
                    
                polygon.find_bottom_and_sideline()
                polygon.disk_cover(n_disk = n_disk)
                polygon.points = polygon.get_coordinates()
                
                    
                    
            
            polygons.append(polygon)
        
        return polygons
    
    def crop_image_by_bbox(self, image, polygon):
        
        word = polygon.text
        box = polygon.points 
        
        rot_angle = None
        match = re.findall(r'\[ROT[0-9]+\]', word)
        if len(match) > 0  :
            rot_angle = int(re.findall(r'\d+', match[0])[0])
        word = polygon.text.replace('[UNK]', '*')
        polygon.text = re.sub(r'[UNK[0-9]+]|[ROT[0-9]+]', '', word)
        if len(box) == 4 :
            w = (int)(np.linalg.norm(box[0] - box[1]))
            h = (int)(np.linalg.norm(box[0] - box[3]))
            width = w
            height = h
            if h > w * 1.5 and len(word) != 1 :
                width = h
                height = w
                # [ROT90]일 때
                if rot_angle == 90 or rot_angle == None or rot_angle == 0 :
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
    
    def inference_pursedo_bboxes(self, image, polygon, viz=False):
        word_bbox = polygon.points
        if len(word_bbox) == 4 :
            word_image, MM = self.crop_image_by_bbox(image, polygon)
        else :
            cropped_white_bg, (og_x, og_y) = self.crop_image_by_bbox(image, polygon)
            word_image = cropped_white_bg
        if word_image is None :
            pass
    
    
        word = re.sub('\[ROT[0-9]+\]','', polygon.text)
        # [UNK{숫자}] 태그 제거
        match = re.findall(r'\[UNK[0-9]+\]', polygon.text)
        if len(match) > 0 :
            #TODO : [ROT]에 대해서 처리
            num = max(int(re.findall(r'\d+', match[0])[0]),1)
            word = '*'*num
        real_word_without_space = word.replace('\s', '').replace('[UNK]', '*')
        num_chars = len(real_word_without_space)
#         input = word_image.copy()
#         scale = 64.0 / input.shape[0]

#         input = cv2.resize(input, None, fx=scale, fy=scale)

        bboxes = []
#         remove_points = []
#         ori_area = cv2.contourArea(polygon)
        
#         for p in range(len()):
#             # attempt to remove p
#             index = list(range(len(word_bbox)))
#             index.remove(p)
#             area = cv2.contourArea(word_bbox[index])
#             if np.abs(ori_area - area) / ori_area < 0.017 and len(word_bbox) - len(remove_points) > 4:
#                 remove_points.append(p)
#         adj_points = np.array([point for i, point in enumerate(word_bbox) if i not in remove_points])

        if not polygon.checked :
            polygon.find_bottom_and_sideline()
            polygon.disk_cover(n_disk = num_chars)
        

#         bottom_pts = textsnake.find_bottom(adj_points)
#         long_edge1, long_edge2 = textsnake.split_long_edges(adj_points, bottom_pts)
#         splited_result1 = textsnake.split_edge_seqence(adj_points, long_edge1, num_chars)
#         splited_result2 = textsnake.split_edge_seqence(adj_points, long_edge2, num_chars)

        bboxes = textsnake.get_char_coordinates(polygon.inner_points1, polygon.inner_points2[::-1])
        bboxes = np.array(bboxes, np.float32)

        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)
        
        if viz :
            img_pil = Image.fromarray(word_image) # 여기서 다시 좌표 조정
            imgdraw = ImageDraw.Draw(img_pil)
            xmin, ymin = word_bbox[:,0].min(), word_bbox[:,1].min()
            for i, bbox in enumerate(bboxes) :
                imgdraw.polygon((bbox - [xmin, ymin]).flatten().tolist(), fill = None, outline = (0,255,0))
                imgdraw.text((bbox[0][0] - xmin, bbox[0][1] - ymin), str(i), (255,0,0))
            plt.figure(figsize = (10,10))
            plt.imshow(np.array(img_pil))
        
            
        
#         if word.startswith('[ROT') :
#             bboxes = np.array(sorted(bboxes, key = lambda x : np.min(x[:,1]), reverse = False))
#         else : # TODO : polygon에 대해 예외 케이스 발생할 수 있음. 근본적인 문제는 affinity point들이 정렬되어 있지 않다는 것
#             bboxes = np.array(sorted(bboxes, key = lambda x : np.min(x[:,0]), reverse = False))
        
        
        if self.watershed_on :
            return bboxes, region_scores, confidence
        else : 
            return bboxes
    

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    
    def load_image_gt_and_confidencemask(self, index):
        imagename = self.images_path[index]
        gt_path = imagename.replace('Train_images','Train_gts').replace('.jpg','.txt').replace('.png','.txt')#os.path.join(self.gt_folder, "%s.txt" % os.path.splitext(imagename)[0])
        polygons = self.load_gt(gt_path)
        image = cv2.imread(imagename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        character_bboxes = []
        word_bboxes = []
        new_words = []
        if len(polygons) > 0:
            for i in range(len(polygons)):
                word = polygons[i].text
                
                if word == '###' or word =='[CBOXF]' or word == '[CBOXT]' or len(word.strip()) == 0:
                    continue
                if self.watershed_on :
                    bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes( image,
                                                                                            polygons[i],
                                                                                            viz=self.viz)
                else :
                    bboxes = self.inference_pursedo_bboxes( image, polygons[i], viz=self.viz)
                
                word_bboxes.append(polygons[i].points)
                word = re.sub('\[ROT[0-9]+\]','', word)
                new_words.append(word)
                
                character_bboxes.append(bboxes)
                
        if len(character_bboxes) != len(new_words) :
            raise ValueError('# bboxes : {}, words : {} '.format(len(character_bboxes), words))
    
        return image, character_bboxes, new_words, word_bboxes, image.shape[:2]

    
    def pull_item(self, index):
#         try : 
        og_image, character_bboxes, words, word_bboxes, og_shape = self.load_image_gt_and_confidencemask(index)

    
        region_scores = np.zeros(og_shape, dtype=np.float32)
        affinity_scores = np.zeros(og_shape, dtype=np.float32)
        orientation_x = np.zeros(og_shape, dtype=np.float32)
        orientation_y = np.zeros(og_shape, dtype=np.float32)
        affinity_bboxes = []
        image_batches = []
        if len(character_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes) # 0-255 range
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_adjust_affinity(region_scores.shape,character_bboxes,words)

            try :
                orientation_x, orientation_y = create_orientation(affinity_bboxes, character_bboxes, region_scores, words)
            except :
                print(index)
                raise ValueError



#         random_transforms = [image, region_scores, affinity_scores, confidence_mask, orientation_x, orientation_y]

#         random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)


#         random_transforms = random_horizontal_flip(random_transforms)
#         random_transforms = random_rotate(random_transforms) # TODO : orientation value changed by rotation

#         cvimage, region_scores, affinity_scores, confidence_mask, orientation_x, orientation_y = random_transforms
#         cvimage = image
        cvimage = cv2.resize(og_image, (self.target_size, self.target_size))
        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
   
        orientation_x = self.resizeGt(orientation_x)
        orientation_y = self.resizeGt(orientation_y)
        ####
        if self.viz:
            self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores/255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores/255).float()
        orientation_x_torch = torch.from_numpy(orientation_x).float()
        orientation_y_torch = torch.from_numpy(orientation_y).float()

        h,w = og_shape
        adj_h = (self.target_size//2)/h
        adj_w = (self.target_size//2)/w

        cropped_word_bboxes, cropped_words = np.zeros((self.str_batch_size, 4,2), np.float32), [self.PAD]*self.str_batch_size

        idx_ls = np.random.choice(range(len(words)), min(len(words),self.str_batch_size), replace = False)
        cnt = 0
        for i, (word, bbox) in enumerate(zip(words, word_bboxes)) :
            ## 추가 (09/30)
            if i in idx_ls :
                if bbox.shape[0] > 4 :
                    rect = cv2.minAreaRect(np.array(bbox, np.float32))
                    bbox = cv2.boxPoints(rect)

    #             word_bboxes[i][:,0] = bbox[:,0]*adj_w
    #             word_bboxes[i][:,1] = bbox[:,1]*adj_h
#                 bbox[:,0] *= adj_w
#                 bbox[:,1] *= adj_h
                cropped_word_bboxes[cnt, :, 0] = bbox[:,0]*adj_w
                cropped_word_bboxes[cnt, :, 1] = bbox[:,1]*adj_h
                cropped_words[cnt] = word

                cnt += 1
#         print('\naccumulate cropped words : ', cnt)
        cropped_word_bboxes[:, :, 1] = np.clip(cropped_word_bboxes[:, :, 1], 0., cvimage.shape[0] - 1)
        cropped_word_bboxes[:, :, 0] = np.clip(cropped_word_bboxes[:, :, 0], 0., cvimage.shape[1] - 1)
        if self.str_cfg.ViTSTR :
            encoded_cropped_words, words_length = self.converter.encode(cropped_words)
        else :
            encoded_cropped_words, words_length = self.converter.encode(cropped_words, batch_max_length = int(self.str_cfg.batch_max_length))



#         image_batches_torch = [torch.from_numpy(image_batch).float() for image_batch in image_batches]

        return image, region_scores_torch, affinity_scores_torch,  orientation_x_torch, orientation_y_torch, cropped_word_bboxes, encoded_cropped_words, words_length

#         except :
#             print(self.images_path[index])
    
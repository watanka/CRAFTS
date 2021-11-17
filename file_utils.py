# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import detection.imgproc
from PIL import Image, ImageDraw



# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes,  font,dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)
        img_pil = Image.fromarray(img)
        imgdraw = ImageDraw.Draw(img_pil)
        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            
            if texts is not None :
                for i, (box, text) in enumerate(zip(boxes, texts)):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) +','+text +'\r\n'
                    # poly = np.array(box).astype(np.int32)
                    # min_x = np.min(poly[:,0])
                    # max_x = np.max(poly[:,0])
                    # min_y = np.min(poly[:,1])
                    # max_y = np.max(poly[:,1])
                    # strResult = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
                    f.write(strResult)

                    poly = poly.reshape(-1, 2)
#                     cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
#                     cv2.putText(img, text, tuple(poly[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.1, color = (0,0,255), thickness= 1)
                    imgdraw.polygon(poly.flatten().tolist(), fill = None, outline = (0,0,255))
                    imgdraw.text(tuple(poly[1]), text,font = font, fill = (0,0,255))
                    
                    ptColor = (0, 255, 255)
                    if verticals is not None:
                        if verticals[i]:
                            ptColor = (255, 0, 0)
            
            else : 
            
                for i, box in enumerate(boxes):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    # poly = np.array(box).astype(np.int32)
                    # min_x = np.min(poly[:,0])
                    # max_x = np.max(poly[:,0])
                    # min_y = np.min(poly[:,1])
                    # max_y = np.max(poly[:,1])
                    # strResult = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
                    f.write(strResult)

                    poly = poly.reshape(-1, 2)
#                     cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                    
                    imgdraw.polygon([poly.reshape((-1,1,2))], fill = None, outline =(0,0,255))

                    ptColor = (0, 255, 255)
                    if verticals is not None:
                        if verticals[i]:
                            ptColor = (255, 0, 0)
        #
        #         if texts is not None:
        #             font = cv2.FONT_HERSHEY_SIMPLEX
        #             font_scale = 0.5
        #             cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
        #             cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        #
        # #Save result image
        cv2.imwrite(res_img_file, np.array(img_pil))

def load_txt(file, delimiter = ',') :
    ## character bbox는 \n\n으로 box별 구분
    coords_ls = []
    with open(file, 'r', encoding = 'utf-8-sig') as f :
        boxes_list = f.read().split('\n\n')
    for boxes in boxes_list :
        if boxes.strip() == '' :
            continue
        char_boxes = boxes.split('\n')
        # char_txt는 라벨이 따로 없다
        charbox_ls =  []
        for charbox in char_boxes :
            if len(char_boxes) == 0 :
                continue
            coords = charbox.split(delimiter)
            coords = [float(c) for c in coords if c != '']
            if len(coords) == 0 :
                continue
            coords = np.array(coords).reshape(-1,2)
            
            charbox_ls.append(coords)
        if len(charbox_ls) != 0 :
            coords_ls.append(np.array(charbox_ls))
            
            
    return coords_ls
            
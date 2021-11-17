import sys
import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageFont
import string
import cv2
from skimage import io
import numpy as np
import detection.craft_utils as craft_utils
import detection.imgproc as imgproc


from config.config import Config

import file_utils
import json
import zipfile
from collections import OrderedDict

from CRAFTS import CRAFTS

import data_utils

from recognition.model import Model
from recognition.collate_fn import RegionAlignCollate
from recognition.utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter


## result folder
result_folder = './wild_scene_result/'
os.makedirs(result_folder, exist_ok = True)

cfg = Config('./config/config.yml')
cfg.training = False
std_cfg = Config(cfg.STD_CONFIG_PATH)
str_cfg = Config(cfg.STR_CONFIG_PATH)


def load_character_list(path) :
    with open(path, 'r') as f :
        character_list = f.read()
        
    return character_list

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

character_list = load_character_list(str_cfg.CHAR_PATH)

if str_cfg.ViTSTR :
    converter = TokenLabelConverter(character_list, str_cfg)
else :
    converter = AttnLabelConverter(character_list)
str_cfg.num_class = len(converter.character)



font = ImageFont.truetype('./assets/batang.ttc', 15)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

crafts = CRAFTS(cfg, std_cfg, str_cfg, device)
if torch.cuda.is_available() :
    crafts.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH)))
else :
    crafts.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH, map_location = 'cpu')))
crafts = crafts.to(device)
padding = RegionAlignCollate(imgH=str_cfg.imgH, imgW=str_cfg.imgW, keep_ratio_with_pad = False)

image_list, _, _ = file_utils.get_files('/home/jovyan/nas/2_public_data/TwinReader/std_data/textdetection_v2_large/AIHUB_wild_scene/Test_images_100/')
                                        #/Hanwha/Test_images/')
                                        

for i, image_path in enumerate(image_list) :
    print("Test image {:d}/{:d}: {:s}".format(i+1, len(image_list), image_path), end='\r')
    image = imgproc.loadImage(image_path)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, std_cfg.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio= std_cfg.MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0)).to(device)
    
    std_out, boxes, polys, preds_str = crafts(x, padding, converter, word_bboxes_batch = None , words_batch = None, words_length_batch = None)
    
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    
    file_utils.saveResult(image_path, image[:,:,::-1], boxes, dirname=result_folder, font = font, texts = preds_str)
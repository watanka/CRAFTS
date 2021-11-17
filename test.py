"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
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

import file_utils
import json
import zipfile
from collections import OrderedDict

from detection.ResUnet import CRAFT



from config import config
import data_utils

from recognition.model import Model
from recognition.collate_fn import RegionAlignCollate
from recognition.utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter

font = ImageFont.truetype('./assets/batang.ttc', 15)

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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


""" For test images in a folder """
image_list, _, _ = file_utils.get_files('/home/jovyan/nas/2_public_data/TwinReader/std_data/textdetection_v2_large/AIHUB_wild_scene/Test_images_100/')
    #'/home/jovyan/nas/3_project_data/TwinReader/Hanwha-Full_text_labeled/1_all_vertical_change/Test_images')
    #'/home/jovyan/nas/2_public_data/TwinReader/std_data/textdetection_v2_large/AIHUB_wild_scene/Test_images_100/')
    #
                                        

result_folder = './test_result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

    
def load_character_list(path) :
    with open(path, 'r') as f :
        character_list = f.read()
        
    return character_list
    
def test_net(STD, STR, image, std_cfg, str_cfg, converter, padding, device, save_crop_batches, image_path, visualize_orientation = True):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, std_cfg.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio= std_cfg.MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
#     if cuda:
#         model = model.to(device)
    x = x.to(device)
    
    # forward pass
    y, feature = STD(x)

    # make score and link map
    score_text = y[0,0,:,:].cpu().data.numpy()
    score_link = y[0,1,:,:].cpu().data.numpy()
    
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    
    plt.imsave('./test_result/score_text.jpg',ret_score_text)
    
    if visualize_orientation :
        score_ori_x = y[0,2,:,:].cpu().data.numpy()
        score_ori_y = y[0,3,:,:].cpu().data.numpy()
        orientation_h, mask = data_utils.scale_orientation(score_ori_x, score_ori_y, score_text)
        # orientation_s
        orientation_s = score_text.copy()
        orientation_s[orientation_s <255*0.2] = 0
        orientation_s = np.uint8(orientation_s)
        # orientation_v
        scale_region = orientation_s.copy()
        scale_region[scale_region<255*0.3] = 0
        orientation_v = scale_region

        hsv = cv2.merge([orientation_h, orientation_s, orientation_v])
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        os.makedirs(os.path.join(result_folder, 'orientation'), exist_ok = True)
        plt.imsave(os.path.join(result_folder, 'orientation', os.path.basename(image_path)), hsv*255.)

    feature_combined = torch.cat([y[:,:2,:,], feature], axis = 1).permute(0,2,3,1)
    
    
#     t0 = time.time() - t0
#     t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, std_cfg.TEXT_THRESHOLD, std_cfg.LINK_THRESHOLD, std_cfg.LOW_TEXT, std_cfg.POLY)

   
    
    STR_input = feature_combined[0].cpu().detach().numpy()
    
#     print('STR input shape :' , STR_input.shape)
#     print('box', boxes[:3])
    
    padded_batch = padding([data_utils.crop_image_by_bbox(STR_input, box, word = '')[0] for box in boxes])
    
    
    if save_crop_batches :
#         ratio_net = 2
        
#         heatmap_features = STR_input[0].detach().numpy()
#         fname = os.path.splitext(os.path.basename(image_path))[0]
#         print(heatmap_features.shape)
#         for idx,box in enumerate(boxes) :
#             xmin, xmax = max(0, int(box[:,0].min())), min(int(box[:,0].max()), y.shape[2])
#             ymin, ymax = max(0, int(box[:,1].min())), min(int(box[:,1].max()), y.shape[1])
#             crop_heatmap = np.float32(heatmap_features[:,ymin:ymax, xmin:xmax])
#             np.save('./test_result/cropped/{}_heatmap_{}.npy'.format(fname, str(idx)), crop_heatmap)
        fname = os.path.splitext(os.path.basename(image_path))[0]
        for i, padded in enumerate(padded_batch) :
            np.save('./test_result/cropped/{}_heatmap_{}.npy'.format(fname, str(i)), padded)
    
#     padded_batch = [p.unsqueeze(0) for p in padded_batch if p is not None]
    
    padded_batch = Variable(padded_batch.type(torch.FloatTensor))
    
    ## TODO
    text_ls = str_test(str_cfg, STR, converter, padded_batch, device)


    
     # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
        
    if save_crop_batches : 
        fname = os.path.splitext(os.path.basename(image_path))[0]
        for idx,box in enumerate(boxes) :
            xmin, xmax = max(0, int(box[:,0].min())), min(int(box[:,0].max()), image.shape[1])
            ymin, ymax = max(0, int(box[:,1].min())), min(int(box[:,1].max()), image.shape[0])
            crop_image = image[ymin:ymax, xmin:xmax, :]
            plt.imsave('./test_result/cropped/{}_image_{}.jpg'.format(fname, str(idx)), crop_image)
    
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text, text_ls


def str_test(str_cfg, str_net, converter, padded_image, device) :
    
    batch_size = padded_image.shape[0]

    length_for_pred = torch.IntTensor([str_cfg.batch_max_length] * batch_size)
    text_for_pred = torch.LongTensor(batch_size, str_cfg.batch_max_length + 1).fill_(0)
    
    str_net = str_net.to(device)
    padded_image = padded_image.to(device)
    length_for_pred = length_for_pred.to(device)
    text_for_pred = text_for_pred.to(device)
    
    
#     if 'CTC' in str_cfg.Prediction:
#         preds = str_net(padded_image, text_for_pred)

#         # Calculate evaluation loss for CTC deocder.
#         preds_size = torch.IntTensor([preds.size(1)] * batch_size)
#         # permute 'preds' to use CTCloss format
# #             if str_cfg.baiduCTC:
# #                 cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
# #             else:
# #                 cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

#         # Select max probabilty (greedy decoding) then decode index to character
#         _, preds_index = preds.max(2)
#         preds_str = converter.decode(preds_index.data, preds_size.data)

#     else:
    preds = str_net(padded_image, text_for_pred, is_train=False)

#             preds = preds[:, :text_for_loss.shape[1] - 1, :]
#             target = text_for_loss[:, 1:]  # without [GO] Symbol
#             cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index, length_for_pred)

    end_token = converter.stop_token
    if end_token is not None :
        preds_str = [pred[:pred.find(end_token)] for pred in preds_str]
            
#         text_ls.append(preds_str)
        
    return preds_str#text_ls
#             labels = converter.decode(text_for_loss[:, 1:], length_for_loss)
        

def test(cfg):
    # load net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cfg.GPU)
    cfg.NUM_GPU = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if cfg.NUM_GPU > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        cfg.WOKRERS = cfg.WORKERS * cfg.NUM_GPU
        cfg.BATCH_SIZE = cfg.BATCH_SIZE * cfg.NUM_GPU

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """
    
    
    
    std_cfg = config.Config(cfg.STD_CONFIG_PATH)
    str_cfg = config.Config(cfg.STR_CONFIG_PATH)
    
    
    character_list = load_character_list(str_cfg.CHAR_PATH)
#     character_list += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㆍㅣ*'
#     character_list += string.printable[:-6]
    
    """ model configuration """
#     if 'CTC' in str_cfg.Prediction:
#         if str_cfg.baiduCTC:
#             converter = CTCLabelConverterForBaiduWarpctc(character_list)
#         else:
#             converter = CTCLabelConverter(character_list)
#     else:
    converter = AttnLabelConverter(character_list)
    str_cfg.num_class = len(converter.character)
    
    
    STD_model = CRAFT(n_classes = std_cfg.NUM_CLASSES)
#     if std_cfg.SAVED_MODEL_PATH :
#         STD_model.load_state_dict(copyStateDict(torch.load(std_cfg.SAVED_MODEL_PATH, map_location='cpu')))
    STR_model = Model(str_cfg)
#     if str_cfg.SAVED_MODEL_PATH :
#         STR_model.load_state_dict(copyStateDict(torch.load(str_cfg.SAVED_MODEL_PATH, map_location='cpu')))
    
    model = nn.ModuleList([STD_model, STR_model])
    
    
    
    
    print('Loading weights from checkpoint {}'.format(cfg.SAVED_MODEL_PATH))
    if cfg.SAVED_MODEL_PATH != '' :
#         if cfg.CUDA:
#             model.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH)))
#         else:
        model.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH, map_location='cpu')))
    
    model.eval()
    
    
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model).to(device)
        cudnn.benchmark = False
        STD = model.module[0]
        STR = model.module[1]
    else :
        STD = model[0]
        STR = model[1]
        STD = STD.to(device)
        STR = STR.to(device)
        
    
    
    

    
    
    t = time.time()
    
    padding = RegionAlignCollate(imgH=str_cfg.imgH, imgW=str_cfg.imgW, keep_ratio_with_pad = False)
    
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, pred_texts = test_net(STD, STR, image, std_cfg, str_cfg, converter, padding = padding, device = device, save_crop_batches = std_cfg.save_crop_batches, image_path = image_path, visualize_orientation = cfg.visualize_orientation)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, font = font, texts = pred_texts)

    print("elapsed time : {}s".format(time.time() - t))


    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='CRAFTS')
    parser.add_argument('--config_path', default = './config/config.yml', type=str, help='path to configuration file')
    parser.add_argument('--cuda', default = True)
    args = parser.parse_args()
    
    cfg = config.Config(args.config_path)
    
    test(cfg)
    
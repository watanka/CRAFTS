import os
import cv2
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import torch.nn.functional as F


class RegionResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        
    def __call__(self, img):
        img = cv2.resize(img, self.size, Image.BICUBIC)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
    
class RegionNormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0) 
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[ :, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img
    
    
    
    
class RegionAlignCollate(object) :
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
    
    def __call__(self, batch):

#         batch = filter(lambda x: x is not None, batch)
        
        if self.keep_ratio_with_pad :
            resized_max_w = self.imgW
            input_channel = batch[0].shape[-1]
            transform = RegionNormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in batch :
                h,w = image.shape[:2]
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = cv2.resize(image, (resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                
            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)    
            
        else :
            transform = RegionResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in batch]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
            
        return image_tensors
        
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

import os, sys, cv2, argparse, time, random, h5py, re, string
from tqdm import tqdm
import datetime
import numpy as np
import scipy.io as scio
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from math import exp
from collections import OrderedDict
from PIL import Image

import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from torch.autograd import Variable
from torch.multiprocessing import Pool, Process, set_start_method

from dataloader_new import Dataset
from CRAFTS import CRAFTS
import detection.coordinates
from detection.mseloss import Maploss
from detection.ResUnet import CRAFT

from recognition.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager, TokenLabelConverter
from recognition.model import Model
from recognition.collate_fn import RegionAlignCollate


from torchutil import *
import data_utils
from config import config

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

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_character_list(path) :
    with open(path, 'r') as f :
        character_list = f.read()
        
    return character_list

if __name__ == '__main__' :
    # argparse
    
    
    parser = argparse.ArgumentParser(description='CRAFTS')
    parser.add_argument('--config_path', default = './config/config.yml', type=str, help='path to configuration file')
    args = parser.parse_args()
    
    cfg = config.Config(args.config_path)
    std_cfg = config.Config(cfg.STD_CONFIG_PATH)
    str_cfg = config.Config(cfg.STR_CONFIG_PATH)
    
    if not cfg.EXP_NAME:
        cfg.EXP_NAME = f'{str_cfg.Transformation}-{str_cfg.FeatureExtraction}-{str_cfg.SequenceModeling}-{str_cfg.Prediction}'
        
        cfg.EXP_NAME += f'-Seed{cfg.SEED}'
        print(cfg.EXP_NAME)

    os.makedirs(f'./saved_models/{cfg.EXP_NAME}', exist_ok=True)
    # vocab / character number configuration
    character_list = load_character_list(str_cfg.CHAR_PATH)
    
    # Seed and GPU setting
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    
    # set up CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cfg.GPU)
    
    cudnn.benchmark = True
    cudnn.deterministic = True
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
    
    print('-'*50)
    print('Data Loading...')
    
    if str_cfg.ViTSTR :
        converter = TokenLabelConverter(character_list, str_cfg)
    else :
        converter = AttnLabelConverter(character_list)
    str_cfg.num_class = len(converter.character)
    
    crafts = CRAFTS(cfg, std_cfg, str_cfg, device)
    
    if cfg.SAVED_MODEL_PATH != '' :
        crafts.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH)))
    if torch.cuda.device_count() > 1 :
        print('Multiple GPUs')
        crafts = torch.nn.DataParallel(crafts).to(device)
    else :
        print('One GPU')
        crafts = crafts.to(device)
    
    crafts.train()
    
    padding = RegionAlignCollate(imgH=str_cfg.imgH, imgW=str_cfg.imgW, keep_ratio_with_pad = False)
    
    real_time=time.time()
    dataloader = Dataset( cfg = cfg,  
                          watershed_on = False, 
                          delimiter = '\t',
                                )
    print('# of Dataset : {}.'.format(len(dataloader.images_path)))
    
    data_loader = torch.utils.data.DataLoader(dataloader,
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=cfg.WORKERS,
                                              drop_last=True,
                                              pin_memory=True)
    
    print("Data Loaded... :: it took {}s.".format( time.time()-real_time))
    
    optimizer = optim.Adam(crafts.parameters(), 
                           lr=float(cfg.LR), 
                           weight_decay=float(cfg.WEIGHT_DECAY))
    
    STR_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device) # 
    STD_criterion = Maploss(ori_lambda = std_cfg.ORIENTATION_WEIGHT).to(device)
    step_index = 0
    
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    print("Training ....")
    for epoch in range(cfg.MAX_EPOCH):
        train_time_st = time.time()
        loss_value = 0
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, cfg.GAMMA, step_index)

        st = time.time()

        for index, (images, gh_label, gah_label, ori_x, ori_y, word_bboxes_batch, words_batch, words_length_batch) in tqdm(enumerate(data_loader)):
            
            
            # Load Variables
            images = Variable(images.type(torch.FloatTensor)).to(device)
            gh_label = gh_label.type(torch.FloatTensor).to(device)
            gah_label = gah_label.type(torch.FloatTensor).to(device)
            gh_label = Variable(gh_label)
            gah_label = Variable(gah_label)
     
            ori_x = ori_x.type(torch.FloatTensor).to(device)
            ori_y = ori_y.type(torch.FloatTensor).to(device)
            ori_x = Variable(ori_x)
            ori_y = Variable(ori_y)

            word_bboxes_batch = word_bboxes_batch.cpu().detach().numpy()

            std_out, preds, target, length = crafts(images, padding, converter, word_bboxes_batch, words_batch, words_length_batch)
            
#             print('--------')
            
            _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
            preds_index = preds_index.view(-1, converter.batch_max_length )
            target_str, pred_str = converter.decode(target, length), converter.decode(preds_index, length)

            rand_idx = random.randint(0, target.shape[0]-1)
            print('\nsample target : {}'.format(target_str[rand_idx]))
            print('sample preds : {}'.format(pred_str[rand_idx]))
#             except :
#                 print('maybe the available cropped images are zero?')
#                 print(target.shape[0])
#                 print(target.shape)
#                 print(preds.shape)

            optimizer.zero_grad()
            
            # calculate loss
            out1 = std_out[:, 0, :, :].to(device)
            out2 = std_out[:, 1, :, :].to(device)
            out3 = std_out[:, 2, :, :].to(device)
            out4 = std_out[:, 3, :, :].to(device)

            STD_loss = STD_criterion(gh_label, gah_label, ori_x, ori_y, out1, out2, out3, out4) 
            if str_cfg.ViTSTR :
                STR_loss = STR_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            else :
                STR_loss = STR_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            print('\nSTD loss : ',STD_loss.item(), end = '\n')
            if torch.isnan(STD_loss):
                torch.save(gh_label, './log/gh_label.pt')
                torch.save(gah_label, './log/gah_label.pt')
                torch.save(ori_x, './log/ori_x.pt')
                torch.save(ori_y, './log/ori_y.pt')
                torch.save(out1, './log/out1.pt')
                torch.save(out2, './log/out2.pt')
                torch.save(out3, './log/out3.pt')
                torch.save(out4, './log/out4.pt')
                
                
                
            print('STR loss : ',STR_loss.item(), end = '\n')
            Total_loss = STD_loss + STR_loss
            Total_loss.backward()
            optimizer.step()
            loss_value += Total_loss.item()
            if index % 100 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 1 batch {} || training loss {}'.format(epoch, index, len(data_loader), et-st, loss_value))
                loss_time = 0
                loss_value = 0
                st = time.time()
                print('Saving state, iter:', epoch)
                torch.save(crafts.state_dict(), os.path.join('./saved_models/', cfg.EXP_NAME, 'CRAFTS' + repr(epoch) + '.pth')) 
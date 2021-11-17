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

from dataloader import Dataset

import detection.coordinates
from detection.mseloss import Maploss
from detection.ResUnet import CRAFT

from recognition.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
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

    # load configuration file .yml
    cfg = config.Config(args.config_path)
    std_cfg = config.Config(cfg.STD_CONFIG_PATH)
    str_cfg = config.Config(cfg.STR_CONFIG_PATH)
    
    # set up save directory
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
#     torch.multiprocessing.set_start_method('spawn')

#     weight_save_dir += '/'+str(len([l for l in os.listdir(weight_save_dir) if l !='.ipynb_checkpoints']))
#     os.makedirs(weight_save_dir, exist_ok = True)
    
    ####load STD model####
    
    STD_model = CRAFT(n_classes = std_cfg.NUM_CLASSES)  
#     if cfg.SAVED_MODEL_PATH == '' :
#         if std_cfg.SAVED_MODEL_PATH != '' :
#             STD_model.load_state_dict(copyStateDict(torch.load(std_cfg.SAVED_MODEL_PATH)))

    print('STD Model Set...')

        
    """ model configuration """
    print('converter set')
    if 'CTC' in str_cfg.Prediction:
        if str_cfg.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(character_list)
        else:
            converter = CTCLabelConverter(character_list)
    else:
        converter = AttnLabelConverter(character_list)
    str_cfg.num_class = len(converter.character)

    ####load STR model####
    STR_model = Model(str_cfg)
    print('model input parameters', str_cfg.imgH, str_cfg.imgW, str_cfg.num_fiducial, str_cfg.input_channel, str_cfg.output_channel,
          str_cfg.hidden_size, str_cfg.num_class, str_cfg.batch_max_length, str_cfg.Transformation, str_cfg.FeatureExtraction,
          str_cfg.SequenceModeling, str_cfg.Prediction)

    
    
    # weight initialization
    for name, param in STR_model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

#     if str_cfg.SAVED_MODEL_PATH != '' and cfg.SAVED_MODEL_PATH == '' :
#         print(f'loading pretrained model from {str_cfg.SAVED_MODEL_PATH}')
#         STR_model.load_state_dict(torch.load(str_cfg.SAVED_MODEL_PATH))
    print('-'*50)
    print('STR Model Set...')
    print('-'*50)
    
    
    real_time=time.time()

    
    model = nn.ModuleList([STD_model, STR_model])
    if cfg.SAVED_MODEL_PATH != '' :
        model.load_state_dict(copyStateDict(torch.load(cfg.SAVED_MODEL_PATH)))
    #TODO : using torch.nn.DataParallel for ModuleList
    
    if torch.cuda.device_count() > 1 :
        print('multiple GPU')
        model = torch.nn.DataParallel(model).to(device)
        STD = model.module[0]
        STR = model.module[1]
    else :
        print('One GPU')
        STD = model[0]
        STR = model[1]
        STD = STD.to(device)
        STR = STR.to(device)

    STD.train()
    STR.train()
    print('-'*50)
    print('CRAFTS Set...')
    print('-'*50)
    
    # handle cropped image
    padding = RegionAlignCollate(imgH=str_cfg.imgH, imgW=str_cfg.imgW, keep_ratio_with_pad = False)
    
    """ STR setup loss """
    STR_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    
    dataloader = Dataset( cfg = cfg, 
                          use_net = False, 
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
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=float(cfg.LR), 
                           weight_decay=float(cfg.WEIGHT_DECAY))
    STD_criterion = Maploss(ori_lambda = std_cfg.ORIENTATION_WEIGHT)
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

        for index, (images, gh_label, gah_label, mask, ori_x, ori_y, word_bboxes_batch, words_batch, words_length_batch) in tqdm(enumerate(data_loader)):
            
            # Load Variables
            images = Variable(images.type(torch.FloatTensor)).to(device)
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).to(device)
            gah_label = Variable(gah_label).to(device)
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).to(device)
            ori_x = ori_x.type(torch.FloatTensor)
            ori_y = ori_y.type(torch.FloatTensor)
            ori_x = Variable(ori_x).to(device)
            ori_y = Variable(ori_y).to(device)
            
            out, feature = STD(images)

            optimizer.zero_grad()
            
            out1 = out[:, 0, :, :].to(device)
            out2 = out[:, 1, :, :].to(device)
            out3 = out[:, 2, :, :].to(device)
            out4 = out[:, 3, :, :].to(device)
       
            STD_loss = STD_criterion(gh_label, gah_label, ori_x, ori_y, out1, out2, out3, out4) 
            
            STR_inputs = torch.cat([out[:,:2,:,:], feature], axis = 1).permute(0,2,3,1).cpu().detach().numpy()  # receive str_input as (batch_size, input_)
            STD_batch_size = images.shape[0]
            
            feature_batch, text_batch, length_batch = [], [], []
            
            for i in range(STD_batch_size) :
                STR_input = STR_inputs[i]
                word_bboxes = word_bboxes_batch[i]
                words = words_batch[i]
                words_length = words_length_batch[i]
                decoded_words = converter.decode(words, words_length)
                
                for word_bbox, word, decoded_word, word_length in zip(word_bboxes, words, decoded_words, words_length) :
                    
                    if word_length != 1  :
                        cropFeature, _ = data_utils.crop_image_by_bbox(STR_input, word_bbox, decoded_word)     
                        
                        feature_batch.append(cropFeature)
                        text_batch.append(word.unsqueeze(0))
                        length_batch.append(word_length.unsqueeze(0))
                            
            pad_batch = padding(feature_batch)
        
            cropped_batch = Variable(pad_batch.type(torch.FloatTensor)).to(device)  
            text_batch = Variable(torch.cat(text_batch).type(torch.LongTensor)).to(device)
            length_batch = Variable(torch.cat(length_batch).type(torch.IntTensor)).to(device)

            batch_size = len(cropped_batch)#image.size(0)
            
            if 'CTC' in str_cfg.Prediction:
                preds = STR(cropped_batch, text_batch)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if str_cfg.baiduCTC:
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                    STR_loss = STR_criterion(preds, text_batch, preds_size, length_batch) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    STR_loss = STR_criterion(preds, text_batch, preds_size, length_batch)

            else:
                preds = STR(cropped_batch, text_batch[:, :-1])  # align with Attention.forward
                target = text_batch[:, 1:]  # without [GO] Symbol
                STR_loss = STR_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            Total_loss = STD_loss + STR_loss
            print('\nSTD loss : ',STD_loss.item(), end = '\n')
            print('STR loss : ',STR_loss.item(), end = '\n')
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
                torch.save(model.state_dict(), os.path.join('./saved_models/', cfg.EXP_NAME, 'CRAFTS' + repr(epoch) + '.pth')) 

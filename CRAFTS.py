import torch.nn as nn
import torch
from detection.ResUnet import CRAFT
import detection.craft_utils as craft_utils

from recognition.model import Model
import data_utils
from config import config
from torch.autograd import Variable
class CRAFTS(nn.Module) :
    def __init__(self, cfg, std_cfg, str_cfg, device) :
        super().__init__()
        self.cfg = cfg
        self.std_cfg = std_cfg
        self.str_cfg = str_cfg
        self.device = device
        self.detection_model = CRAFT(input_channel = 3, n_classes = self.std_cfg.NUM_CLASSES)
        self.recognition_model = Model(self.str_cfg)
        
    def forward(self, x, padding, converter, word_bboxes_batch, words_batch, words_length_batch) :
        
        std_out, feature = self.detection_model(x)
        STR_inputs = torch.cat([std_out[:,:2,:,:], feature], axis = 1).permute(0,2,3,1).cpu().detach().numpy() 
        STD_batch_size = STR_inputs.shape[0]
        
        if self.cfg.training :
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

            pad_batch = padding(feature_batch).to(self.device)

            cropped_batch = Variable(pad_batch.type(torch.FloatTensor)).to(self.device)
            text_batch = Variable(torch.cat(text_batch).type(torch.LongTensor)).to(self.device)
            length_batch = Variable(torch.cat(length_batch).type(torch.IntTensor)).to(self.device)
            
            if self.str_cfg.ViTSTR :
                preds = self.recognition_model(cropped_batch, text_batch)
                target = text_batch
            else :
                preds = self.recognition_model(cropped_batch, text_batch[:, :-1])  # align with Attention.forward
                target = text_batch[:, 1:]

            return std_out, preds, target, length_batch
        
        else :
            ## TODO: apply test.py
            assert std_out.shape[0] == 1
            
            score_text = std_out[0,0,:,:].cpu().data.numpy()
            score_link = std_out[0,1,:,:].cpu().data.numpy()
            boxes, polys = data_utils.getDetBoxes(score_text, score_link, self.std_cfg.TEXT_THRESHOLD, self.std_cfg.LINK_THRESHOLD, self.std_cfg.LOW_TEXT, poly = self.std_cfg.POLY)
            print(polys)
            
            feature_batch = []
            STR_input = STR_inputs[0] 
            for box in boxes :
                cropFeature, _ = data_utils.crop_image_by_bbox(STR_input, box, '')     
                feature_batch.append(cropFeature)
            
            pad_batch = padding(feature_batch).to(self.device)
            batch_size = pad_batch.shape[0]
            
            length_for_pred = torch.IntTensor([self.str_cfg.batch_max_length] * batch_size).to(self.device)
            
            
            if self.str_cfg.ViTSTR :
                text_for_pred = torch.LongTensor(batch_size, self.str_cfg.batch_max_length + 2).fill_(0).to(self.device)
                preds = self.recognition_model(pad_batch, text_for_pred, is_train = False)
            else :
                text_for_pred = torch.LongTensor(batch_size, self.str_cfg.batch_max_length + 1).fill_(0)
                preds = self.recognition_model(pad_batch, text_for_pred, is_train = False).to(self.device)
                
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            end_token = '[s]'
            if end_token is not None :
                if self.str_cfg.ViTSTR :
                    preds_str = [pred[1:pred.find(end_token)] for pred in preds_str]
                else :
                    preds_str = [pred[:pred.find(end_token)] for pred in preds_str]
        #         text_ls.append(preds_str)

            return std_out, boxes, polys, preds_str
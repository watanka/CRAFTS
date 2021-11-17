import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, ori_lambda, use_gpu = True):
        self.ori_lambda = ori_lambda
        super(Maploss,self).__init__()


    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) == 0 :
                    continue
                elif len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss




    def forward(self, gh_label, gah_label, ori_x, ori_y, p_gh, p_gah, p_ori_x, p_ori_y):
        gh_label = gh_label
        gah_label = gah_label
        p_gh = p_gh
        p_gah = p_gah
        
#         scale_ori_x = torch.mul((gh_label> 0.4).float(),ori_x)
#         scale_ori_y = torch.mul((gh_label>0.4).float(),ori_y)
        
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss2 = loss_fn(p_gah, gah_label)
        loss3 = loss_fn(p_ori_x, ori_x)
        loss4 = loss_fn(p_ori_y, ori_y)
        
#         loss_g = torch.mul(loss1, mask)
#         loss_a = torch.mul(loss2, mask)
    
        loss_g = loss1
        loss_a = loss2
        loss_orix = loss3
        loss_oriy = loss4
        
    
        
#         loss_orix = torch.mul(loss3, mask)
#         loss_orix = torch.mul((gh_label>0.4).float(), loss_orix )
#         loss_oriy = torch.mul(loss4, mask)
#         loss_oriy = torch.mul((gh_label>0.4).float(), loss_oriy )
        
        char_loss = self.single_image_loss(loss_g, gh_label)
        affi_loss = self.single_image_loss(loss_a, gah_label)
        orix_loss = self.single_image_loss(loss_orix, ori_x)
        oriy_loss = self.single_image_loss(loss_oriy, ori_y)
        
        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0] + (orix_loss/loss_orix.shape[0] + oriy_loss/loss_oriy.shape[0]) * self.ori_lambda
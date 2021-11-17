import torch.nn as nn
import torch
from torchutil import *

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)
    
class UpBlock(nn.Module) :
    def __init__(self, input_channel, output_channel, upsampling_method = 'conv_transpose') :
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.UpSample = nn.ConvTranspose2d(input_channel, input_channel, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.UpSample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1)
            )

        self.UpConv_Block = nn.Sequential(
                                            nn.Conv2d(input_channel*2, output_channel*2, kernel_size = 1, stride = 1),
                                            nn.BatchNorm2d(output_channel*2),
                                            nn.ReLU(),
                                            nn.Conv2d(output_channel*2, output_channel, kernel_size = 3, stride = 1, padding = 1),
                                            nn.BatchNorm2d(output_channel),
                                            nn.ReLU()            
                                            )
    
    def forward(self, up_x, down_x, return_output = False) :
        x = self.UpSample(up_x)
        x = torch.cat([x, down_x], 1)
        if return_output :
            output_feature = x
        x = self.UpConv_Block(x)
        if return_output :
            return x, output_feature
        else :
            return x
        
class CRAFT(nn.Module) :
    def __init__(self, input_channel = 3, n_classes = 4) :
        super().__init__()
        self.input_channel = input_channel
        self.down_block1 = BottleNeck(self.input_channel, 16)
        self.down_block2 = BottleNeck(64, 64)
        self.down_block3 = BottleNeck(256, 128)
        self.down_block4 = BottleNeck(512, 256)
        self.down_block5 = BottleNeck(1024, 512)
        self.bridge = nn.ConvTranspose2d(2048, 1024, kernel_size=1, stride=1)
        
        self.up_block1 = UpBlock(1024, 512)
        self.up_block2 = UpBlock(512, 256)
        self.up_block3 = UpBlock(256, 64)
        self.up_block4 = UpBlock(64, 32)
        self.last_layer = nn.Sequential(
                                       nn.Conv2d(32,32, kernel_size= 3, stride = 1, padding = 1),
                                       nn.Conv2d(32,32, kernel_size= 3, stride = 1, padding = 1),
                                       nn.Conv2d(32,16, kernel_size= 3, stride = 1, padding = 1),
                                       nn.Conv2d(16,16, kernel_size= 1),
                                       nn.Conv2d(16,n_classes, kernel_size = 1, stride = 1)
                                      )
        
        init_weights(self.down_block1.modules())
        init_weights(self.down_block2.modules())
        init_weights(self.down_block3.modules())
        init_weights(self.down_block4.modules())
        init_weights(self.down_block5.modules())
        init_weights(self.bridge.modules())
        init_weights(self.up_block1.modules())
        init_weights(self.up_block2.modules())
        init_weights(self.up_block3.modules())
        init_weights(self.up_block4.modules())
        init_weights(self.last_layer.modules())

        
    def forward(self, x) :
        pre_pools = dict()
        x = self.down_block1(x)
        pre_pools[f"layer_1"] = x
        x = self.down_block2(x)
        pre_pools[f"layer_2"] = x
        x = self.down_block3(x)
        pre_pools[f"layer_3"] = x
        x = self.down_block4(x) 
        pre_pools[f"layer_4"] = x
        x = self.down_block5(x) 
        x = self.bridge(x)
        x = self.up_block1(x, pre_pools['layer_4'])
        x = self.up_block2(x, pre_pools['layer_3'])
        x = self.up_block3(x, pre_pools['layer_2'])
        x, output_feature = self.up_block4(x, pre_pools['layer_1'], return_output = True)
        x = self.last_layer(x)
             
        return x, output_feature
import torch.optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np

from PIL import Image
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from utils import *
from utilsd_db import *

#################################################################################
#################################################################################
######################## SEGMENTATION CLASS #####################################
class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
     
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    )
            return  block
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding = 1),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding = 1),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        # Decode
        self.conv_decode3   = self.expansive_block(512, 256, 128)
        self.conv_decode2   = self.expansive_block(256, 128, 64)
        self.final_layer    = self.final_block(128, 64, out_channel)
        self.in_channel     = in_channel 
        
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer


#################################################################################
#################################################################################
######################## SEGMENTATION CLASS #####################################
class UNet2Stream(nn.Module):
        
    def __init__(self, in_channel, out_channel):
        super(UNet2Stream, self).__init__()
        
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.u1     = UNet(in_channel=in_channel[0], out_channel= out_channel)
        self.u2     = UNet(in_channel=in_channel[1], out_channel= out_channel)
        self.cf     = torch.nn.Conv2d(kernel_size=3, in_channels = 2, 
                               out_channels=out_channel, padding = 1)
            
    def cat_conv(self, u1, u2, in_channels, out_channels, kernel_size=3):
        x = torch.cat((u1, u2), 1)
        x = self.cf(x) 
        return (x)

    
    def forward(self, im, sl):
        # Encode
        u1      = self.u1(im)
        u2      = self.u2(sl)
        final   = self.cat_conv(u1, u2, self.out_channel*2, self.out_channel)               
        return  final


#################################################################################
#################################################################################
######################## SEGMENTATION CLASS GRU# ################################
class FeatExtract(nn.Module):
    def __init__(self, backend):
        super(FeatExtract, self).__init__()
        #backends = {
        #    'resnet18' : torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        #    }
        #self.net = backends[backend]
        #summary(self.net, (224,224,3))
               
        


#################################################################################
class SoftPsp(nn.Module):
    pass



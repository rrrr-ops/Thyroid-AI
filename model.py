import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models import get_vit_base_patch16_224
from torchvision.models import resnet18, resnet50, resnet34, alexnet, vgg11, vgg16, vgg13, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, densenet121, densenet161, densenet169, densenet201
import numpy as np
from functools import partial


"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import numpy as np
import torch
from torch import nn
from torch.nn import init


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()
        self.conv = self._make_layers(cfg[vgg_name])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           CBAMBlock(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
  
def _add_prefix(state_dict, prefix):
    """
    Remove prefix from parameter names in the state_dict.
    """
    new_state_dict = {}
    for name, param in state_dict.items():
            new_name = prefix + name
            new_state_dict[new_name] = param      
    return new_state_dict

class Vgg13MIL(nn.Module):
    def __init__(self, target_number = None):
        super(Vgg13MIL, self).__init__()
         #####
        
        self.target_number = target_number
        self.vgg13 = vgg13(pretrained=True)
        self.vgg13 = nn.Sequential(*list(self.vgg13.children())[:-2])

        self.L = 768
        self.D = 128
        self.K = 1

        if self.target_number in ['WS', 'CH', 'RAC']:
            t = 3
        else: 
            t = 2

        self.linear = nn.Linear(self.L*self.K, t)
        

        self.attention = nn.Sequential(
            nn.Linear(self.L + 128, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 


        self.attention_vgg13 = nn.Sequential(
            nn.Linear(512, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.linear_vgg13 = nn.Linear(512, 2)

    def forward(self, x):
        H = self.vgg13(x.squeeze(0))  # Use ResNet-18(BN) as the backbone
        H = self.gap(H)
        H = H.squeeze()

        # A = self.attention_vgg13(H)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, H)  # KxL
        # y = self.linear_vgg13(M)

        # max pooling
        M, _ = torch.max(H, dim=0, keepdim=True)  # [1, 512]
        y = self.linear_vgg13(M)      # 分类输出

        # mean pooling
        # M = torch.mean(H, dim=0, keepdim=True) # [1, 512]
        # y = self.linear_vgg13(M)      # 分类输出
        return y


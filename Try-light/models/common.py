# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from timm.models.layers import DropPath

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, colorstr,
                           increment_path, make_divisible, non_max_suppression,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync

from layers import detect_layers


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module(
        'conv',
        nn.Conv2d(in_channels=in_ch,
                  out_channels=out_ch,
                  kernel_size=ksize,
                  stride=stride,
                  padding=pad,
                  bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

#yolo5 old-conv------------------------------------------------------------------------------------------
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self,
#                  c1,
#                  c2,
#                  k=1,
#                  s=1,
#                  p=None,
#                  g=1,
#                  act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1,
#                               c2,
#                               k,
#                               s,
#                               autopad(k, p),
#                               groups=g,
#                               #bias=False)
#                               bias = True)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (
#             act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     # def forward_fuse(self, x):
#     #     return self.act(self.conv(x))
#     def fuse_forward(self, x):
#         return self.act(self.conv(x))


#yolov5 7.0conv -------------------------------------------------------------------------------------------------
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=True)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
#-----------------------------------------------------------------------------------------------------------------


class SPMConv(nn.Module):
    # Standard convolution
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1,
                              c1,
                              k,
                              s,
                              autopad(k, p),
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())
        self.strip_pool1 = StripPooling(c1, (20, 12))
        self.strip_pool2 = StripPooling(c1, (20, 12))
        self.out_layer = nn.Sequential(nn.Conv2d(c1, c1 // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c1 // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(c1 // 2, c2, 1)) 
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.out_layer(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))

#deep width conv---------------------------------------------------------------------------------------------------------

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DW_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(DW_Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, 1)
        self.cv2 = DWConv(c_, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):  # 如果shortcut并且输入输出通道相同则跳层相加
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3DW(nn.Module): 
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3DW, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, 1)
        self.cv2 = DWConv(c1, c_, 1)
        self.cv3 = DWConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[DW_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])  # n个残差组件(Bottleneck)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
#-----------------------------------------------------------------------------------------------------






class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    


class Bottleneck_ds(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.downsample = nn.Conv2d(c1, c2, kernel_size=1, stride=2, bias=False) if c1 != c2 else None
        self.cv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, kernel_size=3, stride=1, padding=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.cv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.cv2(out)
        out = self.bn2(out)

        if self.add:
            out += identity

        out = self.relu(out)

        return out



class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)  
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=0.5)
                                 for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class StripPooling(nn.Module):  #条形磁化
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/2)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))




#替代sppf--------------------------------------------------------------------------------------------------------
class SPPFCSPC(nn.Module):
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
#-------------------------------------------------------------------------------------------------------------


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat([
                x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], 1))
        #return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(
            c1, c1, k, s, act=False), Conv(
                c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size(
        )  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.asff = ASFF()

    def forward(self, x):
        return torch.cat(x, self.d)


# class Add(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, alpha=1):
#         super().__init__()
#         self.a = alpha

#     def forward(self, x):
#         return torch.add(x[0], x[1], alpha=self.a)


class ASFFV5(nn.Module):

    def __init__(self,
                 level,
                 multiplier=1,
                 rfb=False,
                 vis=False,
                 act_cfg=True):
        """
        ASFF version for YoloV5 .
        different than YoloV3
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=1
        256, 128, 64 -> multiplier=0.5
        For even smaller, you need change code manually.
        """
        super(ASFFV5, self).__init__()
        self.level = level
        self.dim = [
            int(1024 * multiplier),
            int(512 * multiplier),
            int(256 * multiplier)
        ]
        # print(self.dim)

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim,
                                       3, 2)

            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim,
                                       3, 2)

            self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)
        elif level == 1:
            self.compress_level_0 = Conv(int(1024 * multiplier),
                                         self.inter_dim, 1, 1)
            self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim,
                                       3, 2)
            self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
        elif level == 2:
            self.compress_level_0 = Conv(int(1024 * multiplier),
                                         self.inter_dim, 1, 1)
            self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim,
                                         1, 1)
            self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)

        # when adding rfb, we use half number of channels to save memory
        compress_c = 8 if rfb else 16
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        # 128, 256, 512
        512, 256, 128
        from small -> large
        """
        x_level_0 = x[2]  # l
        x_level_1 = x[1]  # m
        x_level_2 = x[0]  # s
        # print('x_level_0: ', x_level_0.shape)
        # print('x_level_1: ', x_level_1.shape)
        # print('x_level_2: ', x_level_2.shape)
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2,
                                                     3,
                                                     stride=2,
                                                     padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed,
                                            scale_factor=2,
                                            mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed,
                                            scale_factor=4,
                                            mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed,
                                            scale_factor=2,
                                            mode='nearest')
            level_2_resized = x_level_2

        # print('level: {}, l1_resized: {}, l2_resized: {}'.format(self.level,
        #      level_1_resized.shape, level_2_resized.shape))
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        # print('level_0_weight_v: ', level_0_weight_v.shape)
        # print('level_1_weight_v: ', level_1_weight_v.shape)
        # print('level_2_weight_v: ', level_2_weight_v.shape)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.BatchNorm2d(F_int))

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=None, dnn=True):
        # Usage:
        #   PyTorch:      weights = *.pt
        #   TorchScript:            *.torchscript
        #   CoreML:                 *.mlmodel
        #   TensorFlow:             *_saved_model
        #   TensorFlow:             *.pb
        #   TensorFlow Lite:        *.tflite
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        #   TensorRT:               *.engine
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = [
            '.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '',
            '.mlmodel'
        ]
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (
            suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}'
                             for i in range(1000)]  # assign defaults

        if jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif pt:  # PyTorch
            from models.experimental import attempt_load  # scoped to avoid circular import
            model = attempt_load(weights, map_location=device)
            stride = int(model.stride.max())  # model stride
            names = model.module.names if hasattr(
                model, 'module') else model.names  # get class names
        elif coreml:  # CoreML
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4', ))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(
                ('onnx',
                 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
            import onnxruntime
            session = onnxruntime.InferenceSession(w, None)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            Binding = namedtuple('Binding',
                                 ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(
                    shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data,
                                         int(data.data_ptr()))
            binding_addrs = OrderedDict(
                (n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        else:  # TensorFlow model (TFLite, pb, saved_model)
            if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow *.pb inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(
                        lambda: tf.compat.v1.import_graph_def(gd, name=""),
                        [])  # wrapped
                    return x.prune(
                        tf.nest.map_structure(x.graph.as_graph_element,
                                              inputs),
                        tf.nest.map_structure(x.graph.as_graph_element,
                                              outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd=graph_def,
                                                inputs="x:0",
                                                outputs="Identity:0")
            elif saved_model:
                LOGGER.info(
                    f'Loading {w} for TensorFlow saved_model inference...')
                import tensorflow as tf
                model = tf.keras.models.load_model(w)
            elif tflite:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                if 'edgetpu' in w.lower():
                    LOGGER.info(
                        f'Loading {w} for TensorFlow Lite Edge TPU inference...'
                    )
                    import tflite_runtime.interpreter as tfli
                    delegate = {
                        'Linux':
                        'libedgetpu.so.1',  # install https://coral.ai/software/#edgetpu-runtime
                        'Darwin': 'libedgetpu.1.dylib',
                        'Windows': 'edgetpu.dll'
                    }[platform.system()]
                    interpreter = tfli.Interpreter(
                        model_path=w,
                        experimental_delegates=[tfli.load_delegate(delegate)])
                else:
                    LOGGER.info(
                        f'Loading {w} for TensorFlow Lite inference...')
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(
                        model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt:  # PyTorch
            y = self.model(im) if self.jit else self.model(
                im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.coreml:  # CoreML *.mlmodel
            im = im.permute(
                0, 2, 3,
                1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image':
                                    im})  # coordinates are xywh normalized
            box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
            conf, cls = y['confidence'].max(1), y['confidence'].argmax(
                1).astype(np.float)
            y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)),
                               1)
        elif self.onnx:  # ONNX
            im = im.cpu().numpy()  # torch to numpy
            if self.dnn:  # ONNX OpenCV DNN
                self.net.setInput(im)
                y = self.net.forward()
            else:  # ONNX Runtime
                y = self.session.run(
                    [self.session.get_outputs()[0].name],
                    {self.session.get_inputs()[0].name: im})[0]
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (
                im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        else:  # TensorFlow model (TFLite, pb, saved_model)
            im = im.permute(
                0, 2, 3,
                1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.tflite:
                input, output = self.input_details[0], self.output_details[0]
                int8 = input[
                    'dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., 0] *= w  # x
            y[..., 1] *= h  # y
            y[..., 2] *= w  # w
            y[..., 3] *= h  # h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt or self.engine or self.onnx:  # warmup types
            if isinstance(
                    self.device, torch.device
            ) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(
                    torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self,
                  model,
                  include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'),
                  exclude=())  # copy attributes
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[detect_layers]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(
                    imgs.to(p.device).type_as(p), augment,
                    profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (
            1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], [
        ]  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(
                    requests.get(im, stream=True).raw if str(im).
                    startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(
                    exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose(
                    (1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(
                im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(
                im)  # update
        shape1 = [
            make_divisible(x, int(self.stride.max()))
            for x in np.stack(shape1, 0).max(0)
        ]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0]
             for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(
            p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y,
                                    self.conf,
                                    iou_thres=self.iou,
                                    classes=self.classes,
                                    multi_label=self.multi_label,
                                    max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [
            torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1],
                         device=d) for im in imgs
        ]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n
                       for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self,
                pprint=False,
                show=False,
                save=False,
                crop=False,
                render=False,
                save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(
                            pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(
                                cls)] / self.files[i] if save else None
                            crops.append({
                                'box':
                                box,
                                'conf':
                                conf,
                                'cls':
                                cls,
                                'label':
                                label,
                                'im':
                                save_one_box(box, im, file=file, save=save)
                            })
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(
                im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(
                        f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}"
                    )
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}'
            % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir,
                                  exist_ok=save_dir != 'runs/detect/exp',
                                  mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir,
                                  exist_ok=save_dir != 'runs/detect/exp',
                                  mkdir=True) if save else None
        return self.display(crop=True, save=save,
                            save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[
                x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()
            ] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [
            Detections([self.imgs[i]], [self.pred[i]], self.names, self.s)
            for i in range(self.n)
        ]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])],
            1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
    




#SCConv----------------------------------------------------------------------------------------------------------------------   

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = False
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/torch.sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask   = reweigts>=self.gate_treshold
        noninfo_mask= reweigts<self.gate_treshold
        x_1         = info_mask * gn_x
        x_2         = noninfo_mask * gn_x
        x           = self.reconstruct(x_1,x_2)
        return x
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class SCConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    
#上采样------------------------------------------------------------------------------------------------------------------------------------------
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    


#PSKNet-----------------------------------------------------------------------------------------------------------------------------------------------
# class CAA(nn.Module):
#     """Context Anchor Attention"""
#     def __init__(
#             self,
#             channels: int,
#             h_kernel_size: int = 11,
#             v_kernel_size: int = 11,
#     ):
#         super().__init__()
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)
#         self.conv1 = DWConv(channels, channels, 1, 1, 0)
#         self.h_conv = DWConv(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels)
#         self.v_conv = DWConv(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels)
#         self.conv2 = DWConv(channels, channels, 1, 1, 0)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         return attn_factor

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
        self.CAA = CAA(dim)
        self.conv_caa = nn.Conv2d(dim, dim//2, 1)

    def forward(self, x):
        attn_caa = self.CAA(x)
        attn_caa = self.conv_caa(attn_caa)  
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = attn_caa + attn
        attn = self.conv(attn)
        attn = x * attn

        return attn
    

class LSKlayer(nn.Module):
    def __init__(self, in_channels,):
        super(LSKlayer, self).__init__()
        self.LSK_P = LSKblock(in_channels)

    def LSK(self, x):
        output = self.LSK_P(x)
        return output


    def forward(self, x):
        # Branch 1: Mobile window
        x1 = x

        #Branch 2: Normal convolution
        output_normal = self.LSK(x)
        output_normal = output_normal + x1

        output = output_normal

        return output

class LC(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=4):
        super(LC, self).__init__()
        self.block_size = block_size
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.concat = nn.Conv2d(out_channels * (block_size ** 2), out_channels, kernel_size=1)
        self.normal_conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.normal_conv2 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=7, padding=3)
        # self.normal_conv3 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=11, padding=5)
        self.LSK_P = LSKblock(in_channels)
        #self.sc1 = STConv1(in_channels)

    def mobile_window_branch(self, x):
        block_size = self.block_size
        height, width = x.size(2), x.size(3)
        block_height, block_width = height // block_size, width // block_size
        output_window = torch.zeros_like(x)
        for i in range(block_size):
            for j in range(block_size):
                block = x[:, :, i * block_height : (i + 1) * block_height, j * block_width : (j + 1) * block_width]
                #block_upsampled = self.upsample(block)
                #block_features = F.relu(self.conv1(block_upsampled))
                #block_features = F.relu(self.conv2(block_features))
                #block_features = self.LSK(block_upsampled)
                block_features = self.normal_conv1(block)
                #block_features = self.downsample(block_features)
                output_window[:, :, i * block_height : (i + 1) * block_height, j * block_width : (j + 1) * block_width] = block_features
        return output_window

    def LSK(self, x):
        output = self.LSK_P(x)
        return output
    
    # def STC(self,x):
    #     output = self.sc1(x)
    #     return output

    def forward(self, x):
        # Branch 1: Mobile window
        x1 = x
        # output_window = self.mobile_window_branch(x)
        # output_window = output_window + x1

        #Branch 2: Normal convolution
        output_normal = self.LSK(x)
        output_normal = output_normal + x1

        # output_STC = self.STC(x)
        # output_STC = output_STC + x1

        # Merge branches
        output = output_normal

        return output
    
class DownSamplingLayer(nn.Module):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
    ):
        super(DownSamplingLayer, self).__init__()
        out_channels = in_channels * 2

        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,)

    def forward(self, x):
        return self.down_conv(x)
    
def autopad(k, p=None, dilation=1):
    if p is None:
        p = dilation * (k - 1) // 2
    return p

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SW_Block(nn.Module):
    def __init__(self, x, block_size):
        super(SW_Block,self).__init__()
        self.block_size = block_size
        self.conv1 = DWConv()
        height, width = x.size(2), x.size(3)
        block_height, block_width = height // block_size, width // block_size
        output_window = torch.zeros_like(x)
        for i in range(block_size):
            for j in range(block_size):
                block = x[:, :, i * block_height : (i + 1) * block_height, j * block_width : (j + 1) * block_width]
                #block_upsampled = self.upsample(block)
                #block_features = F.relu(self.conv1(block_upsampled))
                #block_features = F.relu(self.conv2(block_features))
                #block_features = self.LSK(block_upsampled)
                block_features = self.conv1(block)
                #block_features = self.downsample(block_features)
                output_window[:, :, i * block_height : (i + 1) * block_height, j * block_width : (j + 1) * block_width] = block_features
        return output_window

class CAA(nn.Module):
    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11):
        super(CAA, self).__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))      
        return attn_factor

class DKC(nn.Module):
    def __init__(
            self,
            in_channels,
            caa_kernel_size=11,
            block_size = 4,
    ):
        super(DKC, self).__init__()
        out_channels = in_channels
        #hidden_channels = make_divisible(int(out_channels * expansion), 8)
        hidden_channels = in_channels

        self.pre_conv = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)

        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.dw_conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2, groups=hidden_channels)
        self.dw_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3, groups=hidden_channels)
        self.dw_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=9, padding=4, groups=hidden_channels)
        self.dw_conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=11, padding=5, groups=hidden_channels)
        self.pw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        self.post_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.SW = LC(in_channels,out_channels,block_size)

    def forward(self, x):
        attn = x
        x = self.pre_conv(x)
        #z = self.SW(attn)

        y = x.clone()  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        y = self.caa_factor(y)
        y = x * y
        x = x + y
        x = self.post_conv(x)

        return x
    

#starnet----------------------------------------------------------------------------------------------------------------------------------
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



    
    



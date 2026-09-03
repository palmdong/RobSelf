""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .align_extractor_deform import AlignExtractorDeform
from .align_extractor_efficient import AlignExtractorEfficient
from .filter import Filtering


def default_conv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=True)


class ResBlock(nn.Module):
    def __init__(self, conv, n_channel, kernel_size):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            conv(n_channel, n_channel, kernel_size),
            nn.LeakyReLU(),
            conv(n_channel, n_channel, kernel_size),
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RobSelfNetwork(nn.Module):
    def __init__(self, conv=default_conv, scale=None, img_idx=None, output_dir=None):  
        super(RobSelfNetwork, self).__init__()

        self.scale = scale
        self.output_dir = output_dir
        self.eval_idx = 0

        self.branch_guide = nn.Sequential(conv(3, 64, 3), nn.LeakyReLU(), ResBlock(conv, 64, 3), ResBlock(conv, 64, 3))
        self.branch_source = nn.Sequential(conv(1, 64, 3), nn.LeakyReLU(), ResBlock(conv, 64, 3), ResBlock(conv, 64, 3))

        self.align_extractor = AlignExtractorDeform(scale, n_channels=64)
        # self.align_extractor = AlignExtractorEfficient(scale, n_channels=64)

        self.filter = Filtering(scale, n_channels=64, r=4)

        self.branch_pred = nn.Sequential(ResBlock(conv, 64, 1), ResBlock(conv, 64, 1), conv(64, 1, 1))

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules(): 
            if 'align_extractor' in name and 'offset_conv' in name and isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                print(f"Initialized {name} with std=0.001")
            if 'align_extractor' in name and 'mapping' in name and isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                print(f"Initialized {name} with std=0.001")

    def forward(self, guide_img, source_img):
        source_img = F.interpolate(source_img, [guide_img.shape[2],guide_img.shape[3]], mode='bilinear', align_corners=False) 
        guide, source = self.branch_guide(guide_img), self.branch_source(source_img)

        if isinstance(self.align_extractor, AlignExtractorEfficient):
            dis, field, guide_aligned = self.align_extractor(guide, source)
        else:
            offset, guide_aligned = self.align_extractor(guide, source)
            dis = offset

        fake = self.branch_pred(guide_aligned)

        source_enhanced = self.filter(guide_aligned, source)
        pred = self.branch_pred(source_enhanced)

        return source, dis, guide_aligned, fake, pred

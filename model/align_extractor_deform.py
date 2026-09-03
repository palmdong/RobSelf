""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import numpy as np


class GlobalEstimator(nn.Module):
    """
    estimate offset for deformable convolution
    output offset [b, 2*k*k, h, w] for direct use in DeformConv2d
    """

    def __init__(self, scale, n_channels, deform_kernel):
        super(GlobalEstimator, self).__init__()

        self.scale = scale
        self.offset_groups = deform_kernel * deform_kernel

        num = 3 if scale == 2 else 4

        self.encoder = nn.ModuleList()
        for i in range(num):
            if i == 0:
                in_channels = 2 * n_channels
            else:
                in_channels = n_channels
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.LeakyReLU(),
                nn.AvgPool2d(2),
            )
            self.encoder.append(layer)

        self.decoder = nn.ModuleList()
        for i in range(num):
            if i == 0:
                in_channels = n_channels
            else:
                in_channels = 2 * n_channels
            layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_channels,
                    n_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.LeakyReLU(),
            )
            self.decoder.append(layer)

        self.offset_conv = nn.Conv2d(
            n_channels, 2 * self.offset_groups, kernel_size=3, stride=1, padding=1
        )
        self.offset_conv._is_displacement_output = True

    def forward(self, guide, source):
        """
        Args:
            guide: [b, c, h, w]
            source: [b, c, h, w]
        Returns:
            offset: [b, 2*k*k, h, w] for DeformConv2d
        """
        feature = torch.cat([guide, source], dim=1)

        if self.scale == 2:
            # encode
            fea_down1 = self.encoder[0](feature)
            fea_down2 = self.encoder[1](fea_down1)
            fea_down3 = self.encoder[2](fea_down2)
            # decode
            fea_up1 = self.decoder[0](fea_down3)
            fea_up1 = torch.cat([fea_up1, fea_down2], dim=1)
            fea_up2 = self.decoder[1](fea_up1)
            fea_up2 = torch.cat([fea_up2, fea_down1], dim=1)
            fea_up3 = self.decoder[2](fea_up2)
            offset = self.offset_conv(fea_up3)
        else:
            # encode
            fea_down1 = self.encoder[0](feature)
            fea_down2 = self.encoder[1](fea_down1)
            fea_down3 = self.encoder[2](fea_down2)
            fea_down4 = self.encoder[3](fea_down3)
            # decode
            fea_up1 = self.decoder[0](fea_down4)
            fea_up1 = torch.cat([fea_up1, fea_down3], dim=1)
            fea_up2 = self.decoder[1](fea_up1)
            fea_up2 = torch.cat([fea_up2, fea_down2], dim=1)
            fea_up3 = self.decoder[2](fea_up2)
            fea_up3 = torch.cat([fea_up3, fea_down1], dim=1)
            fea_up4 = self.decoder[3](fea_up3)
            offset = self.offset_conv(fea_up4)

        return offset


class AlignExtractorDeform(nn.Module):
    def __init__(self, scale, n_channels, deform_kernel=3):
        super(AlignExtractorDeform, self).__init__()

        self.global_estimate = GlobalEstimator(scale, n_channels, deform_kernel)
        self.deform_conv = DeformConv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=deform_kernel,
            padding=deform_kernel // 2,
        )

    def forward(self, guide, source):
        offset = self.global_estimate(guide, source)
        guide = self.deform_conv(guide, offset)

        return offset, guide

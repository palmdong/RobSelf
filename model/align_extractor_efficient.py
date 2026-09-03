""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def BaseGridGenerator(guide):
    """
    based_grid is generated based on the image or feature to be warped
    [b, h, w, 2], each element represents normalized coordinates [grid_x, grid_y]
    grid_x spans from -1 to 1 along axis x, and is constant along axis y
    grid_y spans from -1 to 1 along axis y, and is constant along axis x
    """
    h, w = guide.shape[2], guide.shape[3]
    x = torch.linspace(-1, 1, w, device=guide.device)
    y = torch.linspace(-1, 1, h, device=guide.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    base_grid = base_grid.unsqueeze(0)
    return base_grid


class GlobalEstimator(nn.Module):
    """
    estimate displacement, dis [b, h, w, 2], from guide to source
    base_grid + dis forms the deformation field for warping
    """

    def __init__(self, scale, n_channels):
        super(GlobalEstimator, self).__init__()

        self.scale = scale
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

        self.mapping = nn.Conv2d(n_channels, 2, kernel_size=3, stride=1, padding=1)
        self.mapping._is_displacement_output = True

    def forward(self, guide, source):
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
            dis = self.mapping(fea_up3).permute(0, 2, 3, 1)
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
            dis = self.mapping(fea_up4).permute(0, 2, 3, 1)

        return dis


class AlignExtractorEfficient(nn.Module):
    def __init__(self, scale, n_channels):
        super(AlignExtractorEfficient, self).__init__()

        self.grid_generator = BaseGridGenerator
        self.global_estimate = GlobalEstimator(scale, n_channels)

    def forward(self, guide, source):
        base_grid = self.grid_generator(guide)
        dis = self.global_estimate(guide, source)
        field = base_grid + dis
        guide_aligned = F.grid_sample(
            guide, field, mode="bilinear", padding_mode="border", align_corners=True
        )

        return dis, field, guide_aligned

""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Filtering(nn.Module):
    def __init__(self, scale, n_channels, r, kernels=(5, 7), eta=0.7):
        super(Filtering, self).__init__()

        self.scale = scale
        self.kernels = kernels
        self.eta = eta
        self.last_kernel_weights = None

        small_kernel, large_kernel = sorted(kernels)
        offset = (large_kernel - small_kernel) // 2
        small_mask = torch.zeros(large_kernel, large_kernel, dtype=torch.bool)
        small_mask[
            offset : offset + small_kernel, offset : offset + small_kernel
        ] = True
        self.register_buffer(
            "small_kernel_mask",
            small_mask.reshape(1, large_kernel**2, 1, 1),
            persistent=False,
        )

        c_mid = n_channels // r
        self.mlp_1 = nn.Conv2d(n_channels, c_mid, 1)
        self.mlp_2 = nn.Conv2d(n_channels, c_mid, 1)
        self.mlp_3 = nn.Conv2d(c_mid, n_channels, 1)

    def compute_importance(self, x):
        grad_x = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        importance = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6).mean(
            dim=1, keepdim=True
        )
        return importance / (importance.mean(dim=(2, 3), keepdim=True) + 1e-6)

    def forward(self, f_guide, f_source):
        f_guide = self.mlp_1(f_guide)
        f_source = self.mlp_2(f_source)

        with torch.no_grad():
            use_large = self.compute_importance(f_source) > self.eta
            # M_imp / (E_hw[M_imp] + eps) > eta
            # This is approximately equivalent to: M_imp > tau,
            # where tau = eta * E_hw[M_imp], and eps is only used for numerical stability.

        if not self.training:
            self.last_kernel_weights = torch.cat(
                [(~use_large).to(f_source.dtype), use_large.to(f_source.dtype)], dim=1
            )
        else:
            self.last_kernel_weights = None

        b, c, h, w = f_source.shape
        kernel_size = max(self.kernels)
        patches = F.unfold(
            f_source, kernel_size=kernel_size, padding=kernel_size // 2
        ).view(b, c, kernel_size**2, h, w)

        logits = (patches * f_guide.unsqueeze(2)).sum(dim=1)
        logits.masked_fill_(
            ~(self.small_kernel_mask | use_large), torch.finfo(logits.dtype).min
        )
        filters = torch.softmax(logits, dim=1)
        filtered = (patches * filters.unsqueeze(1)).sum(dim=2)
        return self.mlp_3(filtered)

""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import numpy as np
import os
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional as F

from .network import RobSelfNetwork


def RobSelf(guide_img, source_img, params=None, target_img=None, img_idx=None, output_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_img = source_img.squeeze()

    # standardization
    guide_img = (guide_img - np.mean(guide_img, axis=(1, 2), keepdims=True)) / np.std(guide_img, axis=(1, 2), keepdims=True)
    source_img_mean, source_img_std = np.mean(source_img), np.std(source_img)
    source_img = (source_img - source_img_mean) / source_img_std
    if target_img is not None:
        target_img = (target_img - source_img_mean) / source_img_std

    # prepare data
    guide_img = torch.from_numpy(guide_img).float().to(device)
    source_img = torch.from_numpy(source_img).float().to(device)
    guide_img = guide_img.unsqueeze(0)
    source_img = source_img.unsqueeze(0).unsqueeze(0)
    if target_img is not None:
        target_img = torch.from_numpy(target_img).float().to(device)

    train_data = torch.utils.data.TensorDataset(guide_img, source_img)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"])

    # setup network
    mynetwork = (RobSelfNetwork(scale=params["scale"], img_idx=img_idx, output_dir=output_dir).train().to(device))
    optimizer = optim.Adam(mynetwork.parameters(), lr=params["lr"])

    total_params = sum(p.numel() for p in mynetwork.parameters())
    trainable_params = sum(p.numel() for p in mynetwork.parameters() if p.requires_grad)
    optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Parameters in Optimizer: {optimizer_params}")

    if params["regress_loss"] == "l1":
        regress_loss = torch.nn.L1Loss()
    elif params["regress_loss"] == "mse":
        regress_loss = torch.nn.MSELoss()
    else:
        print("Unknown loss!")
        return

    # optimization
    with tqdm(range(params["epoch"]), leave=True) as tnr:
        if target_img is not None:
            tnr.set_postfix(RMSE_fake=-1.0, consistency_fake=-1, RMSE=-1, consistency=-1.0)
        else:
            tnr.set_postfix(consistency_fake=-1, consistency=-1.0)

        for epoch in tnr:
            if (epoch + 1) % 5 == 0:
                for p in optimizer.param_groups:
                    p["lr"] *= 0.9998

            mynetwork.train()
            for x, y in train_loader:
                optimizer.zero_grad()

                s_fea, dis, g_a, fake, pred = mynetwork(x, y)
                fake_down, pred_down = F.avg_pool2d(fake, params['scale']), F.avg_pool2d(pred, params['scale'])
                
                consistency_fake, consistency = regress_loss(fake_down, y), regress_loss(pred_down, y)
                total_loss = params["regress_fake"] * consistency_fake + params["regress_pred"] * consistency
            
                total_loss.backward()
                optimizer.step()

    # final prediction
    mynetwork.eval()
    _, _, _, fake, pred = mynetwork(guide_img, source_img)
    fake, pred = fake.squeeze(), pred.squeeze()
    pred = source_img_mean + source_img_std * pred  # de-standardization
    pred = pred.cpu().detach().squeeze().numpy()
    return _, pred

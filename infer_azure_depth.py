""" 
Xiaoyu Dong 
dong@ms.k.u-tokyo.ac.jp
"""

import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.main import RobSelf
from utils.logger import setup_logging


def main():
    logger, log_filename = setup_logging()

    # define parameters
    params = {
        "scale": 2,  # SR factor, 2 or 4
        "img_idxs": [],  # specify images to process, if empty process all
        "regress_loss": "l1",
        "regress_fake": 1,
        "regress_pred": 1,
        "optim": "adam",
        "lr": 0.001,
        "batch_size": 1,
        "epoch": 1000,
    }


    # load data
    scale = params["scale"]
    source_paths = sorted(glob.glob(f"./data/RealMisSR/RGB-depth/complex/x{scale}/*_lr.png", recursive=True))
    guide_paths = sorted(glob.glob(f"./data/RealMisSR/RGB-depth/complex/x{scale}/*_rgb.png", recursive=True,))
    source_imgs = np.array([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in source_paths])
    guide_imgs = np.array([cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB) for p in guide_paths])

    assert len(source_imgs) == len(guide_imgs), "Source and guide images count mismatch!"

    if scale == 2:
        target_paths = sorted(glob.glob(f"./data/RealMisSR/RGB-depth/complex/x{scale}/*_hr.png", recursive=True,))
        target_imgs = np.array([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in target_paths])

    if len(params["img_idxs"]) == 0:
        idxs = np.arange(len(source_imgs))
    else:
        idxs = params["img_idxs"]

    metrics = []

    # start optimization
    for n_image, idx in enumerate(idxs):
        output_dir = f"result_x{scale}/{idx}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"######## Processing {n_image + 1}/{len(idxs)} ########")

        source_img, guide_img = source_imgs[idx], guide_imgs[idx]
        guide_img = guide_img.transpose(2, 0, 1)
        target_img = target_imgs[idx] if scale == 2 else None

        _, pred = RobSelf(
            guide_img=guide_img,
            source_img=source_img,
            params=params,
            target_img=target_img,
            img_idx=idx,
            output_dir=output_dir,
        )

        # save predictions
        if pred.dtype == np.float32 or pred.dtype == np.float64:
            pred = np.clip(pred, 0, 65535).astype(np.uint16)
        else:
            pred = pred
        cv2.imwrite(f"{output_dir}/pred_{idx}.png", pred)
        plt.imsave(f"{output_dir}/pred_{idx}_jet.png", pred, cmap="jet")

        # compute metrics
        import torch
        import pyiqa
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # NIQE
        niqe = pyiqa.create_metric("niqe", device=device)
        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
        pred_tensor = torch.tensor(pred_norm).unsqueeze(0).unsqueeze(0).float()
        try:
            niqe_score = niqe(pred_tensor).item()
        except Exception as e:
            logger.warning(f"NIQE failed: {e}")
            niqe_score = -1

        # rmse
        if target_img is not None:
            rmse = np.sqrt(np.mean((pred - target_img) ** 2)) / 10  # centimeter
            metrics.append([rmse, niqe_score])
            logger.info(f"RMSE: {rmse:.3f}  ---  NIQE: {niqe_score:.3f}\n")
        else:
            metrics.append(niqe_score)
            logger.info(f"NIQE: {niqe_score:.3f}\n")

    # print metrics
    metrics = np.array(metrics)
    if target_img is not None:
        avg_rmse, avg_niqe = np.mean(metrics, axis=0)
        logger.info("======== AVERAGE RESULTS ========")
        logger.info(f"RMSE: {avg_rmse:.3f}")
        logger.info(f"NIQE: {avg_niqe:.3f}")
        logger.info("=================================")
    else:
        avg_niqe = np.mean(metrics, axis=0)
        logger.info("======== AVERAGE RESULTS ========")
        logger.info(f"NIQE: {avg_niqe:.3f}")
        logger.info("=================================")


if __name__ == "__main__":
    main()

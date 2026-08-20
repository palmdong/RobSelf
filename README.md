# RobSelf
PyTorch implementation of "Robust Self-Supervised Cross-Modal Super-Resolution against Real-World Misaligned Observations"  
[[arXiv](https://arxiv.org/abs/2602.18822), [supp](https://drive.google.com/file/d/1fqTYuSY7Qp7PFHiHViZs7y6lz6Bws7ws/view?usp=sharing)] [ECCV paper]

## Updates
**[2026/06/18]** Our paper was accepted to ECCV 2026. See you in Malmö, Sweden.  
**[2026/07/30]** Our collected real-world misaligned dataset (RealMisSR) has been uploaded.

## Overview

<p align="center"> <img src="figs/fig1.png" width="78%"> </p>

<p align="center"> <img src="figs/fig2_model.png" width="78%"> </p>

## Code
TODO.

## RealMisSR Dataset
Our collected real-world misaligned data can be downloaded [here](https://drive.google.com/drive/folders/16gWaPR3mryYSdbAGomWXgK4KGdakFT-4?usp=drive_link). 
- **RGB-depth subset:** 52 groups of simple cases with inherent cross-sensor misalignment; and 60 groups of complex cases with inherent cross-sensor misalignment and random viewpoint variation.
- **RGB-NIR subset:** 50 groups of simple cases with inherent cross-sensor misalignment; and 30 groups of complex cases with inherent cross-sensor misalignment and random object motion.
- Please refer to our paper for more details. 

**License and Usage:** The dataset is provided exclusively for academic research purposes. Any commercial use, redistribution, or modification of the dataset without prior permission is prohibited. Please cite our paper if you find the dataset useful.

## Citation
```
@InProceedings{RobSelf_2026_ECCV,
  author    = {Dong, Xiaoyu and Li, Jiahuan and Cui, Ziteng and Yokoya, Naoto},
  title     = {Robust Self-Supervised Cross-Modal Super-Resolution against Real-World Misaligned Observations},
  booktitle = {ECCV},
  year      = {2026}
}
```

## Contact
Xiaoyu Dong at dong@ms.k.u-tokyo.ac.jp

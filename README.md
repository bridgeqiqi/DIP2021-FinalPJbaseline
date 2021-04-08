# DIP2021-FinalPJbaseline
This repository is for DIP curriculum final project. The baseline model [CSRNet](https://arxiv.org/pdf/1802.10062.pdf) is provided.

# Environment & Folders
- python 3.7.4
- pytorch 1.4.0
- torchvision 0.5.0
- numpy 1.18.5
- tensorboard 2.2.1

This pipeline is a simple framework for crowd counting task including four folders(*datasets*, *losses*, *models*, *optimizers*, *Make_Datasets*) and three files(*main.py*, *test.py*, *train.sh*).

- main.py: The entrance of the main program.
- test.py: Compute the MAE and RMSE metrics among testset images based on your checkpoints.
- train.sh: You can run ```sh ./train.sh```	to launch training.
- datasets: This folder contains dataloaders from different datasets.
- losses: This folder contains different customized loss functions if needed.
- models: This folder contains different models. CSRNet is provided here.
- optimizers: This folder contains different optimzers.
- Make_Datasets: This folder contains density map generation codes.

# Datasets Preparation
- ShanghaiTech PartA and PartB: [download_link](https://pan.baidu.com/s/1nuAYslz)
- UCF-QNRF: [download_link](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- NWPU: [download_link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FNWPU%2DCrowd&originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fc3ViTXA0OHd3SkRpSDBZbFQ4Mk5ZWUJtWTlMMHMtRnByckJjb2FBSmtJMXJ3P3J0aW1lPXlxTUoxbF82MkVn)
- GCC: [download link](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/Eo4L82dALJFDvUdy8rBm6B0BuQk6n5akJaN1WUF1BAeKUA?e=ge2cRg)

The density map generation codes are in Make_Datasets folders.

# NWPU Contest Platform
https://www.crowdbenchmark.com/nwpucrowd.html 

# Reference

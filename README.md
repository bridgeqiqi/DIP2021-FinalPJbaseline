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

After all density maps are generated, run ```ls -R /xx/xxx/xxx/*.jpg > train.txt```, ```ls -R /xx/xxx/xxx/*.jpg > val.txt```, ```ls -R /xx/xxx/xxx/*.jpg > test.txt``` to generate txt files for training, validating and testing.


# Quick Start for Training and Testing

- Training

run ```sh ./train.sh``` or run the following command.
```
python main.py --dataset shanghaitechpa \
--model CSRNet \
--train-files /home/jqgao/workspace/CrowdCounting/TrainingTestingFileLists/ShanghaiTechPartA_full_origin_train.txt \
--val-files /home/jqgao/workspace/CrowdCounting/TrainingTestingFileLists/ShanghaiTechPartA_full_origin_val.txt \
--gpu-devices 4 \
--lr 1e-5 \
--optim adam \
--loss mseloss \
--checkpoints ./checkpoints/demo \
--summary-writer ./runs/demo
```

- Testing

run the following command.
```
python test.py --test-files /home/jqgao/workspace/CrowdCounting/TrainingTestingFileLists/ShanghaiTechPartA_full_origin_test.txt --best-model /home/jqgao/workspace/DIP2021/checkpoints/demo/bestvalmodel.pth
```

# NWPU-Crowd Contest Platform
https://www.crowdbenchmark.com/nwpucrowd.html


# Reference
- [CVPR 2015] [Cross-scene Crowd Counting via Deep Convolutional Neural Networks](https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.pdf)
- [CVPR 2016] [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf) (MCNN, ShanghaiTech Dataset)
- [CVPR 2018] [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_CSRNet_Dilated_Convolutional_CVPR_2018_paper.pdf)
- [ECCV 2018] [Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds](https://openaccess.thecvf.com/content_ECCV_2018/papers/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.pdf) (UCF-QNRF Dataset)
- [CVPR 2019] [Learning from Synthetic Data for Crowd Counting in the Wild](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_From_Synthetic_Data_for_Crowd_Counting_in_the_Wild_CVPR_2019_paper.pdf) (GCC Dataset)
- [TIP 2019] [PaDNet: Pan-Density Crowd Counting](https://arxiv.org/pdf/1811.02805.pdf) 
- [PAMI 2020] [NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization](https://arxiv.org/pdf/2001.03360.pdf) (NWPU-Crowd Dataset)
- [Survey] [CNN-based Single Image Crowd Counting: Network Design, Loss Function and Supervisory Signal](https://arxiv.org/pdf/2012.15685.pdf)
- [Survey] [CNN-based Density Estimation and Crowd Counting: A Survey](https://arxiv.org/pdf/2003.12783.pdf)
- [Survey] [A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation](https://arxiv.org/pdf/1707.01202.pdf)

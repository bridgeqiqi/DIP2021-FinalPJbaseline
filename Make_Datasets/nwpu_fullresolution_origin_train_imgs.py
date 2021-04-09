import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import scipy
import scipy.io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter

import os
import cv2
import math
from tqdm import tqdm


def gaussian_filter_density(gt, pts, r=15, c=15, sigma=4):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = len(pts)

    if gt_count == 0:
        return density

    #     print('=======generating ground truth=========')
    Fixed_H = np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)
    H = Fixed_H
    h, w = gt.shape

    #     print('imageshape: ', gt.shape)

    for i, point in enumerate(pts):
        x = min(w, max(0, abs(int(point[0]))))  # read x?
        y = min(h, max(0, abs(int(point[1]))))  # read y?
        # pixel: (y,x)
        if x >= w or y >= h:
            continue
        x1 = x - int(c / 2)
        x2 = x + int(c / 2)
        y1 = y - int(r / 2)
        y2 = y + int(r / 2)

        dfx1 = 0
        dfx2 = 0
        dfy1 = 0
        dfy2 = 0
        change_H = False
        if x1 <= 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        if y1 <= 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        if x2 >= w:
            dfx2 = x2 - (w - 1)
            x2 = w - 1
            change_H = True
        if y2 >= h:
            dfy2 = y2 - (h - 1)
            y2 = h - 1
            change_H = True

        x1h = dfx1
        y1h = dfy1
        x2h = c - 1 - dfx2
        y2h = r - 1 - dfy2

        if change_H:
            H = np.multiply(cv2.getGaussianKernel(y2h - y1h + 1, sigma),
                            (cv2.getGaussianKernel(x2h - x1h + 1, sigma)).T)

        density[y1:y2 + 1, x1:x2 + 1] += H

        if change_H:
            H = Fixed_H

    #     print('===========done=============')
    return density



def main(image_root_path, image_gt_path, image_save_path, image_gt_save_path):

    images = os.listdir(image_root_path)
    for imagename in images:
        print('current processing: ', imagename)

        r = c = 15
        sigma = 4
        if imagename.find('.jpg')<=0:
            continue
        imageid = int(imagename[:-4])
        if imageid > 3609:
            break
        imagepath = os.path.join(image_root_path, imagename)
        gtpath = os.path.join(image_gt_path, imagename).replace('.jpg','.mat')
        imagesavepath = os.path.join(image_save_path, imagename)
        imagegtsavepath = os.path.join(image_gt_save_path, imagename)

        img = cv2.imread(imagepath)
        mat = scipy.io.loadmat(gtpath)
        points = mat['annPoints']

        density = np.zeros((img.shape[0], img.shape[1]))
        density1 = gaussian_filter_density(density, points, r, c, sigma)

        cv2.imwrite(imagesavepath, img)
        # cv2.imwrite(imagegtsavepath, density1)
        np.save(imagegtsavepath.replace('.jpg', '.npy'), density1)
        print('total actual number: ', len(points))
        print('predict estimates: ', density1.sum())




    print('==>saving finish.')


if __name__ == '__main__':
    image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part1'
    image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train/'
    image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train_gt'

    main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)

    image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part2'
    image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train/'
    image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train_gt'

    main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)

    image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part3'
    image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train/'
    image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train_gt'

    main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)

    image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part4'
    image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train/'
    image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Train_gt'

    main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)



    # image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part5a'
    # image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    # image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Test/'
    # image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Test_gt'
    #
    # main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)
    #
    # image_root_path = '/mnt/pami14/jqgao/NWPU-Crowd/images_part5b'
    # image_gt_path = '/mnt/pami14/jqgao/NWPU-Crowd/mats'
    # image_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Test/'
    # image_gt_save_path = '/mnt/pami23/jqgao/CrowdCountingDatasets/NWPU/fullresolution/origin/Test_gt'
    #
    # main(image_root_path, image_gt_path, image_save_path, image_gt_save_path)
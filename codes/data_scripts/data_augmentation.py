import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import sys
sys.path.append("..")
import data.util as util

# path: [1, 800]
def check_path(path):
    index = path.rfind('/') + 1
    if path[index: index + 2] != '08' and path[index: index + 2] != '09':
        return True
    elif path[index: index + 4] == '0800':
        return True
    else:
        return False

for scale in [1, 0.9, 0.8, 0.7, 0.6]:
    GT_folder = '/path/to/you/project/dataset/DIV2K/DIV2K_train_HR'
    save_GT_folder = '/path/to/you/project/classsr/datasets/DIV2K800_scale/GT'
    for i in [save_GT_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    img_GT_list = util._get_paths_from_images(GT_folder)
    for path_GT in img_GT_list:
        if check_path(path_GT) == False:
            continue
        img_GT = cv2.imread(path_GT)
        img_GT = img_GT
        # imresize

        rlt_GT = util.imresize_np(img_GT, scale, antialiasing=True)
        print(str(scale) + "_" + os.path.basename(path_GT))
        
        if scale == 1:
            cv2.imwrite(os.path.join(save_GT_folder,os.path.basename(path_GT)), rlt_GT)
        else:
            cv2.imwrite(os.path.join(save_GT_folder, str(scale) + "_" + os.path.basename(path_GT)), rlt_GT)

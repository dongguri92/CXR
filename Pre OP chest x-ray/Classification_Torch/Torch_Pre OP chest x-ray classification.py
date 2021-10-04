import torch
# torch gpu가져오기
torch.cuda.get_device_name(0)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

!pip install pydicom

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import pydicom

from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

normal_path = "/Normal"
LULz_path = "/TB/LULz"
RULz_path = "TB/RULz"

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, normal_path, LULz_path, RULz_path):
        self.normal_path = normal_path
        self.LULz_path = LULz_path
        self.RULz_path = RULz_path

        self.img_h = 1024
        self.img_w = 1024

        # path
        self.normal_name = os.listdir(normal_path)
        self.TB_LULz_name = os.listdir(LULz_path)
        self.TB_RULz_name = os.listdir(RULz_path)

        self.normal_paths = []
        self.TB_LULz_paths = []
        self.TB_RULz_paths = []

        for i in range(len(self.normal_name)):
            path = self.normal_path + '/' + self.normal_name[i]
            self.normal_paths.append(path)

        for j in range(len(self.TB_LULz_name)):
            path = self.LULz_path + '/' + self.TB_LULz_name[j]
            self.TB_LULz_paths.append(path)

        for k in range(len(self.TB_RULz_name)):
            path = self.RULz_path + '/' + self.TB_RULz_name[k]
            self.TB_RULz_paths.append(path)

        # img path
        self.TB_paths = self.TB_LULz_paths + self.TB_RULz_paths
        self.train_paths = self.normal_paths + self.TB_paths

        self.num_trains = len(self.train_paths)

    def __len__(self):
        return self.num_trains

    def __getitem__(self):

        # labels
        labels_N = [np.array([1,0]) for _ in range(len(self.normal_paths))]
        labels_TB = [np.array([0,1]) for _ in range(len(self.TB_paths))]

        train_labels = labels_N + labels_TB

        # shuffle
        s = np.arange(len(self.train_paths))
        np.random.shuffle(s)

        train_paths = np.array(self.train_paths)
        train_paths = train_paths[s]
        train_paths = list(train_paths)

        train_labels = np.array(train_labels)
        train_labels = train_labels[s]

        img_list = []

        # IMG load
        for l in range(len(train_paths)):
            img = pydicom.read_file(train_paths[l])
            arr = img.pixel_array
            re_img = cv2.resize(arr, (self.img_h, self.img_w), interpolation=cv2.INTER_AREA)
            img_list.append(re_img)

        return (img_list, train_labels)

# 우선 데이터불러오는 class까지 만듦
# --> dataloader만든 후 efficientnet으로 학습해보기

import torch
# torch gpu가져오기
torch.cuda.get_device_name(0)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

!pip install pydicom
!pip install efficientnet_pytorch

import glob
import os
import os.path as osp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import PIL
from PIL import Image
import cv2
import pydicom

from efficientnet_pytorch import EfficientNet

# transform
class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(0.485, 0.229)])

    def __call__(self, img):
        return self.data_transform(img)

# dataset
class CXRDataset(data.Dataset):
    def __init__(self, normal_path, LULz_path, RULz_path, transform=None):
        self.transform = transform

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

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        # index번째 CXR 로드
        img_path = self.train_paths[index]
        img_origin = pydicom.read_file(img_path)
        img_arr = img_origin.pixel_array
        img_re = cv2.resize(img_arr, (1024,1024), interpolation=cv2.INTER_LINEAR)
        img = img_re.astype(np.float32)

        # transforms
        img_transformed = self.transform(img)

        # label
        if "TB" in img_path:
            label = 1
        else:
            label = 0

        return img_transformed, label

# Dataset

normal_path = "Normal"
LULz_path = "LULz"
RULz_path = "RULz"

train_dataset = CXRDataset(
    normal_path, LULz_path, RULz_path, transform=ImageTransform()
)

# 미니 배치 크기 지정
batch_size = 32

# 데이터 로더 작성
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True
)

# 우선 데이터불러오는 class까지 만듦
# --> dataloader만든 후 efficientnet으로 학습해보기

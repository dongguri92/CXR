import torch

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
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

import PIL
from PIL import Image
import cv2
import pydicom

from efficientnet_pytorch import EfficientNet

class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, img):
        return self.data_transform(img)

class CXRDataset(data.Dataset):
    def __init__(self, normal_path, cancer_path, transform=None):
        self.transform = transform

        self.normal_path = normal_path
        self.cancer_path = cancer_path

        self.img_h = 512
        self.img_w = 512

        # path
        self.normal_name = os.listdir(normal_path)
        self.cancer_name = os.listdir(cancer_path)

        self.normal_paths = []
        self.cancer_paths = []

        for i in range(len(self.normal_name)):
            path = self.normal_path + '/' + self.normal_name[i]
            self.normal_paths.append(path)

        for j in range(len(self.cancer_name)):
            path = self.cancer_path + '/' + self.cancer_name[j]
            self.cancer_paths.append(path)

        # img path
        self.train_paths = self.normal_paths + self.cancer_paths
        self.train_paths = self.train_paths[:2060]

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        # index번째 CXR 불러오기
        img_path = self.train_paths[index]
        img_origin = pydicom.read_file(img_path)
        img_arr = img_origin.pixel_array
        del img_origin
        img_re = cv2.resize(img_arr, (self.img_h, self.img_w), interpolation=cv2.INTER_LINEAR)
        del img_arr
        img = img_re.astype(np.float32)
        del img_re

        # transform
        img_transformed = self.transform(img)
        del img

        # label
        if 'lung_cancer' in img_path:
            label = 1
        else:
            label = 0

        return img_transformed, label

# Dataset

normal_path = "CXR_cancer/normal"
cancer_path = "CXR_cancer/lung_cancer"

train_dataset = CXRDataset(
    normal_path, cancer_path, transform = ImageTransform()
)

# 미니 배치 크기 지정
batch_size = 16

# 데이터 로더 작성
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle = True
)

# 모델
class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.FC = nn.Linear(1000, 2)


    def forward(self, x):

        x = F.relu(self.conv2d(x))

        x = F.relu(self.efficientnet(x))

        x = torch.sigmoid(self.FC(x))
        return x

# 모델 선언
model = Efficientnet()

model = Efficientnet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

print("DEVICE: ", DEVICE)
print("MODEL: ", model)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data_, target_) in enumerate(train_loader):
        data, target = data_.to(DEVICE), target_.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 8 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))
        del data_
        del target_
        del data
        del target
        torch.cuda.empty_cache()

EPOCHS = 25
for epoch in range(1, EPOCHS + 1):
    train(model, train_dataloader, optimizer, epoch)

# 모델 저장
torch.save(model.state_dict(), "/efficientnet.pth")
print("saved.....")

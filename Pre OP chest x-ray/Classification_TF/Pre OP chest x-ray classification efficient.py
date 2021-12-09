!pip install pydicom

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import pydicom

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models

from tensorflow.keras.applications import EfficientNetB7
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Label

Label_name = ['normal', 'Old TB']


# name

normal_name = os.listdir("CXR/data/Normal")

TB_LULz_name = os.listdir("CXR/data/TB/LULz")

TB_RULz_name = os.listdir("CXR/data/TB/RULz")

# path

normal_path = []
TB_LULz_path = []
TB_RULz_path = []

for i in range(len(normal_name)):
    path = "CXR/data/Normal/" + normal_name[i]
    normal_path.append(path)

for j in range(len(TB_LULz_name)):
    path = "CXR/data/TB/LULz/" + TB_LULz_name[j]
    TB_LULz_path.append(path)

for k in range(len(TB_RULz_name)):
    path = "CXR/data/TB/RULz/" + TB_RULz_name[j]
    TB_RULz_path.append(path)


# img path

TB_path = TB_LULz_path + TB_RULz_path

train_paths = normal_path + TB_path

# labels

labels_N = [np.array([1,0]) for _ in range(len(normal_path))]
labels_TB = [np.array([0,1]) for _ in range(len(TB_path))]

train_labels = labels_N + labels_TB

# shuffle
s = np.arange(len(train_paths))
np.random.shuffle(s)

train_paths = np.array(train_paths)
train_paths = train_paths[s]
train_paths = list(train_paths)

train_labels = np.array(train_labels)
train_labels = train_labels[s]
train_labels = list(train_labels)

# IMG load

img_list = []

for i in range(len(train_paths)):
    img = pydicom.read_file(train_paths[i])
    arr = img.pixel_array
    re_img = cv2.resize(arr, (600,600), interpolation=cv2.INTER_AREA)
    img_list.append(re_img)

# img sample

img_ex = img_list[0]
print("shape : ", img_ex.shape)
plt.imshow(img_ex)
plt.show()

# np.array

train_imgs = np.array(img_list)
train_imgs = np.expand_dims(train_imgs, axis = 3)
train_labels = np.array(train_labels)

(img_rows, img_cols, input_dims) = (600,600,1)
input_shape = (img_rows, img_cols, input_dims)

nb_classes = 2

# hyperparameter
num_ch = 64

lr = 0.001
batch_size = 1
epochs = 40

# efficient net

efficient_net = EfficientNetB7(
    input_shape=(600,600,3),
    include_top=False,
    pooling='max'
)

model = tf.keras.Sequential()
model.add(keras.Input(shape=(600,600,1))) # channel이 1인 x ray image를 넣기 위해 추가한 layer
model.add(layers.Conv2D(3, 5, padding='same', activation='relu'))
model.add(efficient_net)
model.add(Dense(units = 1024, activation='relu'))
model.add(Dense(units = 1024, activation = 'relu'))
model.add(Dense(units = 2, activation='softmax'))
model.summary()

adam = optimizers.Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.01, amsgrad = False)

model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )


# fit
history = model.fit(train_imgs, train_labels, epochs = epochs, batch_size = batch_size)

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

# Label

Label_name = ['normal', 'Old TB']

# Path

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
    re_img = cv2.resize(arr, (1024,1024), interpolation=cv2.INTER_AREA)
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

(img_rows, img_cols, input_dims) = (1024,1024,1)
input_shape = (img_rows, img_cols, input_dims)

nb_classes = 2

# hyperparameter
num_ch = 64

lr = 0.0001
batch_size = 4
epochs = 40

# resnet

def model(input_shape, nb_classes):

	#input
    img_input = layers.Input(shape = input_shape)

    #block1
    x = layers.Conv2D(num_ch, (7,7), strides = (2,2), padding = 'same', name = 'block1_Conv2D')(img_input)
    x = layers.MaxPooling2D((3,3), strides = (2,2), name = 'block1_maxpooling')(x)
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block2
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block2_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "blaock2_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block3
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block3_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch, (3,3), padding = 'same', name = "block3_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block4
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block4_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block4_maxpooling")(x)
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block4_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block5
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block5_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*2, (3,3), padding = 'same', name = "block5_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block6
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block6_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block6_maxpooling")(x)
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block6_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block7
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block7_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*4, (3,3), padding = 'same', name = "block7_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block8
    x = layers.Conv2D(num_ch*8, (3,3), padding = 'same', name = "block8_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block8_maxpooling")(x)
    x = layers.Conv2D(num_ch*8, (3,3), padding = 'same', name = "block8_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block9
    x = layers.Conv2D(num_ch*8, (3,3), padding = 'same', name = "block9_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*8, (3,3), padding = 'same', name = "block9_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block10
    x = layers.Conv2D(num_ch*16, (3,3), padding = 'same', name = "block10_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block10_maxpooling")(x)
    x = layers.Conv2D(num_ch*16, (3,3), padding = 'same', name = "block10_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block11
    x = layers.Conv2D(num_ch*16, (3,3), padding = 'same', name = "block11_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*16, (3,3), padding = 'same', name = "block11_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block12
    x = layers.Conv2D(num_ch*32, (3,3), padding = 'same', name = "block12_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block12_maxpooling")(x)
    x = layers.Conv2D(num_ch*32, (3,3), padding = 'same', name = "block12_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block13
    x = layers.Conv2D(num_ch*32, (3,3), padding = 'same', name = "block13_Conv2D_1")(input_layer)
    x = layers.Conv2D(num_ch*32, (3,3), padding = 'same', name = "block13_Conv2D_2")(x)
    x = layers.add([x, input_layer])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #block14
    x = layers.Conv2D(num_ch*64, (3,3), padding = 'same', name = "block14_Conv2D_1")(input_layer)
    shortcut = layers.MaxPooling2D((2,2), name = "block14_maxpooling")(x)
    x = layers.Conv2D(num_ch*64, (3,3), padding = 'same', name = "block14_Conv2D_2")(shortcut)
    x = layers.add([x, shortcut])
    x = layers.BatchNormalization()(x)
    input_layer = layers.Activation('relu')(x)

    #flatten
    x = layers.Flatten(name = "flatten")(input_layer)
    x = layers.Dense(8192, name = "flatten_1")(x)
    x = layers.Dense(1024, name = "flatten_2")(x)
    x = layers.Dense(nb_classes, activation = 'sigmoid', name = "prediction")(x)

	#model
    model = models.Model(img_input, x)

    return model

model = model(input_shape, nb_classes)

# optimizer
adam = optimizers.Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.01, amsgrad = False)

# compile
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

# fit
history = model.fit(train_imgs, train_labels, epochs = epochs, batch_size = batch_size)

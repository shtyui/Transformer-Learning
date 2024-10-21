# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:08:52 2023

@author: s205
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# GoogLeNet模型
def inception_module(x, filters):
    conv1 = Conv2D(filters=filters[0], kernel_size=(1,1), padding='same', activation='relu')(x)
    conv3 = Conv2D(filters=filters[1], kernel_size=(3,3), padding='same', activation='relu')(x)
    conv5 = Conv2D(filters=filters[2], kernel_size=(5,5), padding='same', activation='relu')(x)
    maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    return tf.keras.layers.concatenate([conv1, conv3, conv5, maxpool], axis=-1)

input_layer = Input(shape=(28, 28, 1))
x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = inception_module(x, [64, 128, 32])
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.summary()

# 编译和训练模型
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy:', accuracy)
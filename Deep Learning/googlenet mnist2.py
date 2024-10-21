# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:49:21 2023

@author: s205
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data preprocessing
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Inception module
def inception_module(x, n_filters):
    tower_1 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(x)
    tower_3 = Conv2D(n_filters, (5, 5), padding='same', activation='relu')(tower_3)
    tower_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    tower_4 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(tower_4)
    return concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

# Build the GoogLeNet model for MNIST
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = inception_module(x, 64)
x = inception_module(x, 64)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = inception_module(x, 64)
x = inception_module(x, 64)
x = inception_module(x, 64)
x = inception_module(x, 128)
x = inception_module(x, 128)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = GlobalAveragePooling2D()(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.summary()

# Compile and train the model
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy:', accuracy)
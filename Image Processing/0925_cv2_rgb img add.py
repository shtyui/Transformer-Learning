# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:07:28 2023

@author: USER
"""

import numpy as np
import cv2
import random

# 定义图像的宽度、高度和通道数
width, height, channels = 640, 480, 3

# 创建两张随机彩色影像
random_image1 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
random_image2 = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

# 将两张图像相加
result_image = cv2.add(random_image1, random_image2)

# 显示原始图像和结果图像
cv2.imshow('Random Image 1', random_image1)
cv2.imshow('Random Image 2', random_image2)
cv2.imshow('Result Image', result_image)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

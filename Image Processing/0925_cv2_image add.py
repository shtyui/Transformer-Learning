# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:18:01 2023

@author: USER
"""

import numpy as np
import cv2 as cv

img1 = cv.imread('seacow.png')
img2 = cv.resize(img1,(512,488),interpolation=cv.INTER_CUBIC)
cv.imshow('img1',img2)
img3 = cv.imread('wombat.jpg')
img4 = cv.resize(img3,(512,488),interpolation=cv.INTER_CUBIC)
cv.imshow('img2',img4)


dst = cv.addWeighted(img2, 0.7, img4, 0.3, 0)
cv.imshow('dst',dst)

cv.waitKey(0)
cv.destroyAllWindows()
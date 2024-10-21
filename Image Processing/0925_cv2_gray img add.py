# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:58:40 2023

@author: USER
"""
import numpy as np 
import cv2 as cv
import random

img1 = np.random.randint(0,255,(200,300),dtype=np.uint8)
cv.imshow("image1",img1)
img2 = np.random.randint(0,255,(200,300),dtype=np.uint8)
cv.imshow("image2",img2)
img3 = cv.add(img1,img2)

cv.imshow("add image3",img3)

cv.waitKey(0)
cv.destroyAllWindows()
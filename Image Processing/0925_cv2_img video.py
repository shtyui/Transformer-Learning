# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:11:47 2023

@author: USER
"""

import numpy as np
import cv2 as cv

img1 = cv.imread('seacow.png')
img2 = cv.resize(img1, (512, 488), interpolation=cv.INTER_CUBIC)
cv.imshow('img1', img2)
img3 = cv.imread('wombat.jpg')
img4 = cv.resize(img3, (512, 488), interpolation=cv.INTER_CUBIC)
cv.imshow('img2', img4)

for i in range(0,3):
    for alpha in range(0,100):
     
        dst = cv.addWeighted(img2, alpha*0.01, img4, 1 - (alpha*0.01), 0)
    
        cv.imshow('dst', dst)
    
        cv.waitKey(20)
    for alpha in range(0,100):
     
        dst = cv.addWeighted(img4, alpha*0.01, img2, 1 - (alpha*0.01), 0)
    
        cv.imshow('dst', dst)
    
        cv.waitKey(20)
    
cv.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:02:17 2023

@author: USER
"""

import cv2 as cv
import numpy as np
import sys
import matplotlib
img = cv.imread('wombat.jpg')
print('imgae shape=',img.shape)
if img is None:
    sys.exit("Cloud not read the image")

cv.imshow("Display window",img)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("output.png",img)

cv.destroyAllWindows()
print('saving image is done')

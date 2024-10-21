# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:26:52 2023

@author: USER
"""

import cv2 as cv
import numpy as np
import random


r = np.random.randint(0, 255, (200,300), dtype=np.uint8)
g = np.random.randint(0, 255, (200,300), dtype=np.uint8)
b = np.random.randint(0, 255, (200,300), dtype=np.uint8)

img = cv.merge([r,g,b])

cv.imshow("create image",img)
cv.waitKey(0)
cv.destroyAllWindows()
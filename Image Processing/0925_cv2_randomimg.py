# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:25:35 2023

@author: USER
"""

import numpy as np 
import cv2 as cv
import random

x1 = np.random.randint(10,20)
print("回應值是10(含)至20(含)的隨機數")
print(x1)
print("-"*70)
print("回傳一維陣列10個元素,值是1(含)至5(不含)的隨機數")

x2 = np.random.randint(1,5,10)
print(x2)
print("-"*20)
print("回傳單3*5陣列, 值是0(含)至10(不含)的隨機數")

x3 = np.random.randint(10,size=(3,5))
print(x3)

b = np.random.randint(0,255,(200,300),dtype=np.uint8)
g=b
r=b
img = cv.merge([b,g,r])

cv.imshow("create image",img)
cv.waitKey(0)
cv.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:26:48 2023

@author: child
"""

import cv2
import numpy as np 
src = cv2.imread('789.jpg')
cv2.imshow('src',src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',src_gray)
ret, dst_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY)
counters, hierachy = cv2.findContours(dst_binary,
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

box = cv2.minAreaRect(counters[0])
print(f'轉換前的矩形頂角 = \n {box}')
points = cv2.boxPoints(box)
points = np.int0(points)
print(f'轉換後的矩形頂角 = \n {points}')
dst = cv2.drawContours(src, [points], 0, (0, 255, 0), 2)
cv2.imshow('dst',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
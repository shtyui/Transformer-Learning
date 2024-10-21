# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:02:27 2023

@author: child
"""

import cv2
import numpy as np 
src = cv2.imread('789.jpg')
cv2.imshow('src',src)

ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    
lower = np.array([80, 135, 85])
upper = np.array([255, 180, 135])
    
mask = cv2.inRange(ycrcb, lower, upper)
    
res = cv2.bitwise_and(src,src,mask = mask)

#cv2.imshow('mask',mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 確保至少有一個輪廓
if len(contours) > 0:
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(src, [box], 0, (0, 255, 0), 2)
# 計算最優擬合橢圓
if len(contours) > 0:
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(src, ellipse, (0, 0, 255), 2)
cv2.imshow('Face Image', src)           
cv2.waitKey(0)
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:57:37 2023

@author: s205
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    lower = np.array([80, 135, 85])
    upper = np.array([255, 180, 135])
    
    mask = cv2.inRange(ycrcb, lower, upper)
    
    res = cv2.bitwise_and(frame,frame,mask = mask)
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    
    
cap.release()
cv2.destroyAllWindows()

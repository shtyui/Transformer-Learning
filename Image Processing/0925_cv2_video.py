# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:58:48 2023

@author: USER
"""

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("111.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break
    
    cv.imshow('frame',frame)
    if cv.waitKey(30) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:22:08 2023

@author: child
"""
import cv2
img = cv2.imread('123.jpg')

cv2.imshow('BRGimage', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

im1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
cv2.imshow('YCBCRimage',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('YCBCRimage',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

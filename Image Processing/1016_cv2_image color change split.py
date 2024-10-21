# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:57:48 2023

@author: child
"""

import cv2
img = cv2.imread('123.jpg')

b,g,r = cv2.split(img)
cv2.imshow('BRGimage', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('B',b)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('g',g)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('r',r)
cv2.waitKey(0)
cv2.destroyAllWindows()

im1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
b1,g1,r1 = cv2.split(im1)
cv2.imshow('YCBCRimage',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('B',b1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('g',g1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('r',r1)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b2,g2,r2 = cv2.split(im2)
cv2.imshow('YCBCRimage',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('B',b2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('g',g2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('r',r2)
cv2.waitKey(0)
cv2.destroyAllWindows()
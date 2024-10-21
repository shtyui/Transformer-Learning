# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:14:14 2023

@author: child
"""

import cv2 
src = cv2.imread('789.png')
cv2.imshow('src', src)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',src_gray)
ret, dst_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY)

contours, hierachy = cv2.findContours(dst_binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

dst = cv2.drawContours(src, contours, -1, (0, 255, 0), 5)
cv2.imshow('drawed_contour',dst)
K = len(contours)
print('number of contours=', K)
print('-'*70)
for i in range(K):
    print(f'contour_index=(i)')
    print(f'number of contour points={len(contours[i])}')
    print(f'contour_shape=(contours[i].shae')
    print('-'*70)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
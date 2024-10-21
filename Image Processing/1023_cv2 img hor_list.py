# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:35:55 2023

@author: child
"""


import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('789.jpg')
cv2.imshow('123.jpg',img)
print('-'*70)

print('original image',img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print('binaryimg',bw.shape)
bw = np.array(bw)

hor_hist = np.sum(bw == 255, axis = 1)

plt.plot(hor_hist)
plt.title('Horizontal Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Count')
plt.show()

ver_hist = np.sum(bw == 255, axis = 0 )

plt.plot(ver_hist)
plt.title('Vertical Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Count')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
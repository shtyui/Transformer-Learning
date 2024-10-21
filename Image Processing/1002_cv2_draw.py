# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:54:10 2023

@author: USER
"""

import numpy as np
import cv2

# 創建一個白色背景的畫布
canvas = cv2.imread('456.jpg')  # 白色背景

print('imgae shape=',canvas.shape)
# 在畫布上繪製不同的幾何圖形
h, w, c = canvas.shape
print('h',h/2)
center_x = w//2
center_y = h//2
# 畫一條直線
cv2.line(canvas, (5, 5), (w-5, h-5), (0, 0, 255), 5)  # 起點、終點、顏色、線寬

cv2.line(canvas, (5, h-5), (w-5, 5), (0, 0, 255), 5)  # 起點、終點、顏色、線寬
# 畫一個矩形
cv2.rectangle(canvas, (center_x-10, center_y-10), (center_x+10, center_y+10), (0, 255, 0), 2)  # 左上角、右下角、顏色、線寬

# 畫一個圓形
cv2.circle(canvas, (center_x, center_y), 50, (255, 0, 0), 2)  # 圓心、半徑、顏色、-1表示填充

# 顯示畫布
cv2.imshow("Geometry Shapes", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:45 2023

@author: child
"""

import cv2
import numpy as np

# 載入人臉影像
src = cv2.imread('789.jpg')

# 轉換為HSV色彩空間
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# 定義在HSV中膚色的下界和上界
lower = np.array([0, 20, 70], dtype=np.uint8)
upper = np.array([20, 255, 255], dtype=np.uint8)

# 創建一個遮罩以提取膚色
mask = cv2.inRange(hsv, lower, upper)

# 將遮罩應用到原始影像
res = cv2.bitwise_and(src, src, mask=mask)

# 將結果二值化以創建二值圖像
_, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# 在二值圖像中找到輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 對最大的輪廓擬合橢圓（您可以根據需求修改此部分）
if len(contours) > 0:
    largest = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(largest)

    # 在原始影像上畫出橢圓
    cv2.ellipse(src, ellipse, (0, 255, 0), 2)

if len(contours) > 0:
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 顯示帶有矩形框的影像
cv2.imshow('face', src)

cv2.waitKey(0)
cv2.destroyAllWindows()


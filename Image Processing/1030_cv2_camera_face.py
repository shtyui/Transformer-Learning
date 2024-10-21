# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:44:15 2023

@author: child
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 20, 75])
    upper_blue = np.array([20, 190, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    contours,  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到包围矩形
    min_rect = None
    for contour in contours:
        if len(contour) >= 4:  # 至少需要4个点来计算最小包围矩形
            min_rect = cv2.minAreaRect(contour)

    if min_rect is not None:
        # 从旋转包围矩形中获取坐标、宽度和高度
        center, size, angle = min_rect
        width, height = size
        x, y = center
        angle = -angle  # 修正角度

        print("最小包围矩形左上角坐标:", (x - width / 2, y - height / 2))
        print("宽度:", width)
        print("高度:", height)
        print("旋转角度:", angle)


        rect_points = cv2.boxPoints(min_rect).astype(int)
        cv2.drawContours(frame, [rect_points], 0, (0, 255, 0), 2)  # 在图像上绘制最小包围矩形


    ellipse = None
    for contour in contours:
        if len(contour) >= 5:  # 至少需要5个点来进行椭圆拟合
            ellipse = cv2.fitEllipse(contour)

    if ellipse is not None:
        center, axes, angle = ellipse
        major_axis, minor_axis = axes
        x, y = center
        angle = -angle  # 修正角度

        print("椭圆中心坐标:", center)
        print("主轴长度:", major_axis)
        print("副轴长度:", minor_axis)
        print("椭圆倾斜角度:", angle)


        cv2.ellipse(frame, ellipse, (255, 0, 0), 2)  # 在图像上绘制拟合的椭圆

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
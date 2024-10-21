# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:18:21 2023

@author: s205
"""

import cv2

# 使用OpenCV的Haar级联分类器加载人脸和眼睛检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 读取图片
image = cv2.imread('789.jpg')

# 将图像转换为灰度图像以进行人脸和眼睛检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # 在检测到的人脸周围绘制矩形
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 在人脸区域内检测眼睛
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    for (ex, ey, ew, eh) in eyes:
        # 在检测到的眼睛周围绘制矩形
        cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

# 显示图片
cv2.imshow('Face and Eye Detection', image)
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:50:49 2023

@author: s205
"""

import cv2

# 使用OpenCV的Haar级联分类器加载人脸和眼睛检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 从摄像头中读取帧
    ret, frame = cap.read()

    if not ret:
        break

    # 将帧转换为灰度图像以进行人脸和眼睛检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 在检测到的人脸周围绘制矩形
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 在人脸区域内检测眼睛
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # 在检测到的眼睛周围绘制矩形
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('Face and Eye Detection', frame)

    # 如果按下 'q' 键，退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源和关闭所有窗口
cap.release()
cv2.destroyAllWindows()

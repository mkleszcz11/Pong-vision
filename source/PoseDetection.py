import cv2
import time
import numpy as np
import requests
import imutils
import PoseDetectionModule as PDM

cap = cv2.VideoCapture(0)

detector = PDM.PoseDetector()

while True:
    success, img = cap.read()
    img, landmark_lst = detector.find_pose(img)
    print(f"Landmark index and XY position: {landmark_lst}")
    cv2.imshow("Android_cam", img)
    cv2.waitKey(1)

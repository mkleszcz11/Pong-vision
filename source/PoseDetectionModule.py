import cv2
import time
import numpy as np
import requests
import imutils
import mediapipe as mp

class PoseDetector():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils


    def find_pose(self, img, draw=True):
        results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        landmark_lst = []
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark): 
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id in [0, 30, 16, 12]:
                    landmark_lst.append([id, cx, cy])
                    # print(f"Landmark index: {id} - XY [{cx}, {cy}]")
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return img, landmark_lst

def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img, landmark_lst = detector.find_pose(img)
        print(f"Landmark index and XY position: {landmark_lst}")
        cv2.imshow("Android_cam", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
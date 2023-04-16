import cv2
import time
import numpy as np
import json
import requests
import imutils
import PoseDetectionModule as PDM

cap = cv2.VideoCapture(0)

detector = PDM.PoseDetector()

prev_time = 0
curr_time = 0


neutral_pose = []
neutral_pose_counter = 0
nose_mean_x = 0
nose_mean_y = 0
wrist_mean_x = 0
wrist_mean_y = 0
heel_mean_x = 0
heel_mean_y = 0
shoulder_mean_x = 0
shoulder_mean_y = 0
result = 0
while True:
    neutral_pose_counter += 1
    success, img = cap.read()
    img, landmark_lst = detector.find_pose(img)
    if neutral_pose_counter < 10 and landmark_lst[0] and landmark_lst[1] and landmark_lst[2] and landmark_lst[3]:
        neutral_pose.append(landmark_lst)
        nose_mean_x += landmark_lst[0][1]
        nose_mean_y += landmark_lst[0][2]
        wrist_mean_x += landmark_lst[1][1]
        wrist_mean_y += landmark_lst[1][2]
        heel_mean_x += landmark_lst[2][1]
        heel_mean_y += landmark_lst[2][2]
        shoulder_mean_x += landmark_lst[3][1]
        shoulder_mean_y += landmark_lst[3][2]
    print(f"Landmark index and XY position: {landmark_lst}")
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    nose_mean_x /= 10
    nose_mean_y /= 10
    wrist_mean_x /= 10
    wrist_mean_y /= 10
    heel_mean_x /= 10
    heel_mean_y /= 10
    shoulder_mean_x /= 10
    shoulder_mean_y /= 10
    if landmark_lst and abs(landmark_lst[1][2]-wrist_mean_y) < 50:  # pozycja neutralna
        result = 1
    elif landmark_lst and landmark_lst[2][2]<landmark_lst[0][2]: # ręce nad głową
        result = 2
    elif landmark_lst and abs(landmark_lst[2][2]-landmark_lst[3][2]) < 50: # kucanie
        result = 3
    else:
        result = 1
    print(f"Result: {result}")
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
    cv2.imshow("Android_cam", img)
    #print(*neutral_pose)

    # Define the JSON data
    data = {
        "id": 4,
        "racketDirection": result,
    }

    # Encode the data as a JSON string
    json_data = json.dumps(data)

    # Define the URL to send the data to
    url = "http://localhost:5001/"

    # Send the JSON data as part of an HTTP POST request
    response = requests.post(url, data=json_data)

    cv2.waitKey(1)



import numpy as np
import cv2
import dlib
import time
import os

PREDICTOR_PATH = "models"+os.sep+"facial_landmark_extraction"+os.sep+"dlib_facial_landmark"+os.sep+"shape_predictor_68_face_landmarks.dat"

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))


class Dlib_facial_landmark():
    def __init__(self):
        print("[INFO] Loading facial landmarks model  ", PREDICTOR_PATH)
        self.model =  dlib.shape_predictor(PREDICTOR_PATH)
    def get_facial_landmarks(self,image,faces,time_passed=0):
        result= {}
        # Read the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Draw a rectangle around the faces
        facial_landmarks = []
        for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Converting the OpenCV rectangle coordinates to Dlib rectangle
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                start = time.time()

                landmarks = np.matrix([[p.x, p.y]
                                       for p in self.model(image, dlib_rect).parts()])
                result["time"] = time_passed + time.time() - start

                landmarks_display = landmarks[:]
                res_landmarks= []
                for idx, point in enumerate(landmarks_display):
                    pos = (point[0, 0], point[0, 1])
                    res_landmarks.append(pos)
                    cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=-1)
                facial_landmarks.append(res_landmarks)
        result["image"] = image
        result["landmarks"] = facial_landmarks
        return result
import cv2
import os
import numpy as np
import time
MODEL_PATH= "models"+os.sep+"face_detection"+os.sep+"haar_cascade"+os.sep+"haarcascade_frontalface2.xml"

class HaarCascade_detector():
    def __init__(self):
        print("[INFO] loading model...")
        self.detector = cv2.CascadeClassifier(MODEL_PATH)
    def detect_faces(self,img):
        return self.detector.detect_faces()
    def get_faces(self,image_path,crop=False):
        real_image = cv2.imread(image_path)
        result = {}
        image = real_image.copy()
        boxes = []
        cropped = []
        start = time.time()
        faces = self.detector.detectMultiScale(image)  # result
        result["time"] = time.time() - start
        # to draw faces on image
        for box in faces:
            x, y, w, h = box
            boxes.append((x,y,w,h))
            x1, y1 = x + w, y + h
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
            if crop:
                # 'box': [x, y, width, height], crop_img = img[y:y+h, x:x+w]
                cropped.append(image[y:y1, x:x1])

        result["cropped"] = cropped
        result["boxes"] = boxes
        result["image"] = image

        return result

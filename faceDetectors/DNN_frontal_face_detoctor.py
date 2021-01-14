import cv2
import numpy as np
import os
from os.path import dirname, join
import time
filename = join(dirname(__file__), "chatbot.txt")

MODEL_PATH= "models"+os.sep+"face_detection"+os.sep+"DNN_face_module"
PROTO_TXT = MODEL_PATH+os.sep+"deploy.prototxt.txt"
MODEL= MODEL_PATH+os.sep+"res10_300x300_ssd_iter_140000.caffemodel"

class DNN_frontal_face_detoctor():
    def __init__(self):
        self.detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)


    def detect_faces(self,img):
        return self.detector.detect_faces()
    def get_faces(self,image_path,crop=False):
        real_image = cv2.imread(image_path)
        result = {}
        result["image"] = real_image
        image = real_image.copy()
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 117.0, 123.0))
        self.detector.setInput(blob)
        start= time.time()
        faces = self.detector.forward()
        result["time"] = time.time() - start
        boxes = []
        cropped = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.9:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                boxes.append((x,y,x1-x,y1-y))
                cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
                if crop:
                    # 'box': [x, y, width, height], crop_img = img[y:y+h, x:x+w]
                    cropped.append(image[y:y1, x:x1])
        result["cropped"] = cropped
        result["boxes"] = boxes
        result["image"] = image

        return result

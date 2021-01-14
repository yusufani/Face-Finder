import cv2
from mtcnn import MTCNN

import time
class Mtcnn_detector():
    def __init__(self):
        self.detector = MTCNN()
    def detect_faces(self,img):
        return self.detector.detect_faces()
    def get_faces(self,images_path,crop=False):

        real_img = cv2.imread(images_path)
        img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        #img = cv2.imread(images_path)
        result = {}

        start = time.time()
        faces = self.detector.detect_faces(img)
        result["time"] = time.time() - start
        cropped = []
        boxes = []
        for face in faces:
            x, y, width, height = face['box']
            keypoints = face['keypoints']
            cv2.rectangle(real_img,
                          (x, y),
                          (x + width, y + height),
                          (0, 155, 255),
                          2)
            boxes.append((x,y,width,height))
            '''
            cv2.circle(real_img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
            '''
            if crop:
                # 'box': [x, y, width, height], crop_img = img[y:y+h, x:x+w]
                cropped.append( real_img[y:y+height, x:x+width] )
        result["image"] = real_img
        result["cropped"] = cropped
        result["boxes"] = boxes
        return result
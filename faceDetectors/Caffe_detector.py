import cv2
import os
import numpy as np
MODEL_PATH= "models"+os.sep+"face_detection"+os.sep+"caffe"+os.sep+"VGG_ILSVRC_19_layers"
PROTO_TXT = MODEL_PATH+os.sep+"deploy.prototxt.txt"
MODEL= MODEL_PATH+os.sep+"VGG_ILSVRC_19_layers.caffemodel"
class Caffe_detector():
    def __init__(self):
        print("[INFO] loading model... ",MODEL)
        self.detector = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)
    def detect_faces(self,img):
        return self.detector.detect_faces()
    def get_faces(self,image_path,crop=False):
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        print("[INFO] computing object detections...")
        self.detector.setInput(blob)
        print("saddas")
        detections = self.detector.forward()
        print("Detections received now calculating ")
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
        '''
        real_img = cv2.imread(images_path)
        img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        #"img = cv2.imread(images_path)
        faces = self.detector.detect_faces(img)
        cropped = []
        result = {}
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
            cv2.circle(real_img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(real_img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
            if crop:
                # 'box': [x, y, width, height], crop_img = img[y:y+h, x:x+w]
                cropped.append( real_img[y:y+height, x:x+width] )
        result["original_image"] =real_img
        result["cropped"] = cropped
        result["boxes"] = boxes
        return result
        '''
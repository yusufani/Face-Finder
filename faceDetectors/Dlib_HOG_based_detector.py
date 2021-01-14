import dlib
import cv2
import time
class Dlib_HOG_based_detector():
    def __init__(self):
        print("[INFO] loading model...")
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self,img):
        return self.detector.detect_faces()
    def get_faces(self,image_path,crop=False):
        real_image = cv2.imread(image_path)
        result = {}
        image = real_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # This model works on gray scale image
        boxes = []
        cropped = []
        start = time.time()
        faces = self.detector(gray, 1) # result
        result["time"] = time.time() - start

        # to draw faces on image
        for box in faces:
            x = box.left()
            y = box.top()
            x1 = box.right()
            y1 = box.bottom()

            boxes.append((x,y,x1-x,y1-y))

            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
            if crop:
                # 'box': [x, y, width, height], crop_img = img[y:y+h, x:x+w]
                cropped.append(image[y:y1, x:x1])

        result["cropped"] = cropped
        result["boxes"] = boxes
        result["image"] = image
        return result

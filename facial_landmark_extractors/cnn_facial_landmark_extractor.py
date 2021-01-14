from tensorflow import keras

import tensorflow as tf
import numpy as np
import cv2
import dlib
import os
import time
# https://github.com/yinguobing/cnn-facial-landmark


MODEL_PATH = "models" + os.sep + "facial_landmark_extraction" + os.sep + "cnn_facial_lanmark" + os.sep + "pose_model"



class Cnn_facial_landmark:
    def __init__(self):
        print("[INFO] Loading facial landmarks model  ", MODEL_PATH)
        self.model = keras.models.load_model(MODEL_PATH)

    def get_facial_landmarks(self, image, faces,time_passed=0):
        result= {}
        facial_landmarks = []
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            offset_y = int(abs((h) * 0.1))
            box_moved = self.move_box((x, y, x + w, y + h), [0, offset_y])
            facebox = self.get_square_box(box_moved)
            face_img = image[facebox[1]: facebox[3],
                       facebox[0]: facebox[2]]
            try:
                face_img = cv2.resize(face_img, (128, 128))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                start = time.time()
                # # Actual detection.
                predictions = self.model.signatures["predict"](
                    tf.constant([face_img], dtype=tf.uint8))
                result["time"] = time_passed + time.time() - start
                # Convert predictions to landmarks.
                marks = np.array(predictions['output']).flatten()[:136]
                marks = np.reshape(marks, (-1, 2))

                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                marks = marks.astype(np.uint)
                color = (0, 255, 0)
                landmarks = []
                for mark in marks:
                    landmarks.append((mark[0], mark[1]))
                    cv2.circle(image, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)
                facial_landmarks.append(landmarks)
            except:
                facial_landmarks.append("null")
                result["time"] = "null"

        result["image"] = image
        result["landmarks"] = facial_landmarks
        return result

    def get_square_box(self, box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
        # Bug fix for negative values
        if left_x<0 :left_x = 0
        if top_y<0 :top_y = 0
        if right_x < 0: right_x = 0
        if bottom_y<0 :bottom_y = 0

        return [left_x, top_y, right_x, bottom_y]

    def move_box(self, box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

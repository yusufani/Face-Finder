import argparse
from faceDetectors.Mtcnn_detector import Mtcnn_detector
from faceDetectors.Caffe_detector import Caffe_detector
from faceDetectors.Haar_cascade_detector import HaarCascade_detector
from faceDetectors.Dlib_HOG_based_detector import Dlib_HOG_based_detector
from faceDetectors.DNN_frontal_face_detoctor import DNN_frontal_face_detoctor
from utils import *
from facial_landmark_extractors.dlib_facial_lanmark_extractor import *
from facial_landmark_extractors.cnn_facial_landmark_extractor import Cnn_facial_landmark
import pandas as pd

supported_img_types = ("png", "jpg", "PNG", "JPG", "JPEG", "jpeg")
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str,
                        help='Can be folder or image ')
    parser.add_argument('--detectors', nargs="+", default=[], type=str)
    parser.add_argument('--facial_landmarks', nargs="+", default=[], type=str)
    parser.add_argument('--live', type=bool, default=False,help="Not implemented yet")
    parser.add_argument('--crop', type=bool, default=False)
    parser.add_argument('--save_res', type=bool, default=False)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument('--show_images', type=bool, default=False)
    parser.add_argument('--save_facial_landmarks', type=bool, default=True)
    args = parser.parse_args()
    return args

def initialize_detectors(detectors_list):
    detectors = {}
    if len(detectors_list) == 0:  # get all detectors
        detectors["mtcnn"] = Mtcnn_detector()
        # detectors["caffe_berkeley_fine_tuned"] = Caffe_detector()
        detectors["haar_cascade"] = HaarCascade_detector()
        detectors["dlib_hog_based"] = Dlib_HOG_based_detector()
        detectors["dnn_frontol_face"] = DNN_frontal_face_detoctor()
    else:
        for detector in detectors_list:
            if detector == "mtcnn":
                detectors["mtcnn"] = Mtcnn_detector()
            elif detector == "caffe_berkeley_fine_tuned":
                detectors["caffe_berkeley_fine_tuned"] = Caffe_detector()
            elif detector == "haar_cascade":
                detectors["haar_cascade"] = HaarCascade_detector()
            elif detector == "dlib_hog_based":
                detectors["dlib_hog_based"] = Dlib_HOG_based_detector()
            elif detector == "dnn_frontol_face":
                detectors["dnn_frontol_face"] = DNN_frontal_face_detoctor()
            else:
                assert NotImplemented(detector, "is  Not Implemented")
    return detectors

def get_faces_from_detectors(detectors, image_path, crop=False):
    results = {}
    for detector_name, detector in detectors.items():
        results[detector_name] = detector.get_faces(image_path, crop)
    return results

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def initialize_facial_landmarks(facial_landmarks_list):
    facial_landmarks = {}
    if len(facial_landmarks_list) == 0:  # get all detectors
        facial_landmarks["dlib_facial_landmark"] = Dlib_facial_landmark()
        facial_landmarks["cnn_facial_landmark"] = Cnn_facial_landmark()
    else:
        for facial_landmark in facial_landmarks_list:
            if facial_landmark == "dlib_facial_landmark":
                facial_landmarks["dlib_facial_landmark"] = Dlib_facial_landmark()
            elif facial_landmark == "cnn_facial_landmark":
                facial_landmarks["cnn_facial_landmark"] = Cnn_facial_landmark()
            else:
                assert NotImplemented(facial_landmark, "is  Not Implemented")
    return facial_landmarks


def main(args):
    if not args.live:
        if os.path.isdir(args.images):
            print("Found images")
            images_paths = {os.path.join(args.images, i): None for i in os.listdir(args.images) if
                            i.endswith(supported_img_types)}
        else:
            images_paths = {args.images: None}

        detectors = initialize_detectors(args.detectors)
        facial_landmarks = initialize_facial_landmarks(args.facial_landmarks)

        landmarks_csv = []

        for idx, image_path in enumerate(images_paths.keys()):
            faces = get_faces_from_detectors(detectors, image_path, args.crop)
            print("Image : ", image_path)
            for detector, res in faces.items():
                print(res["boxes"])
                for name, extractor in facial_landmarks.items():
                    print("Detector : ", detector, " Facial landmark" , name)
                    if len(res["boxes"] )>= 1 :
                        data= extractor.get_facial_landmarks(res["image"].copy(), res["boxes"])
                    else:
                        data = {}
                        data["landmarks"] = ["null"]
                        data["time"]= ["null"]
                    if args.save_facial_landmarks:
                        landmarks_csv.append( {"path":image_path, "detector":detector , "facial_landmarker": name , "landmarks":data["landmarks"][0],"time":data["time"],'face_boxes': res["boxes"]} )
                    if data["landmarks"][0] != "null":
                        if args.save_res:
                            os.makedirs(args.output, exist_ok=True)
                            save_image(args.output, image_path, data["image"],extra_info="_detecor_{}_facial_landmark_{}".format(detector,name))
                        if args.show_images:
                            cv2.imshow("image" + image_path, data["image"])
                            cv2.waitKey()
                        if args.crop:
                            for idx, image in enumerate(res["cropped"]):
                                if args.save_res:
                                    save_image(args.output, image_path, image["image"], extra_info="cropped_" + str(idx))
                                if args.show_images:
                                    cv2.imshow("Cropped Image" + image_path, image["image"])
        if args.save_facial_landmarks:
            output_name =  "".join(args.images.split(".")[:-1]) + "result.csv" if "." in args.images else args.images+ "result.csv"
            pd.DataFrame(landmarks_csv).to_csv(output_name, index=False)
    else:
        assert NotImplemented("Live option not implemented yet ")

if __name__ == '__main__':
    args = get_args()
    main(args)
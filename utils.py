import cv2
import os
import ntpath

def save_image(output,filename,image,extra_info=""):
    filename ="".join(ntpath.basename(filename).split(".")[:-1])
    cv2.imwrite(os.path.join(output,filename+extra_info+".png"), image)



import time
import cv2
import json
from PIL import Image
from matplotlib import pyplot as plt
import mediapipe as mp
import numpy as np
from parse import JPPNet_parsing
font = cv2.FONT_HERSHEY_SIMPLEX
from rembg import removeBgAPI , getMaskAPI
import os

import numpy as np
mp_pose = mp.solutions.pose


def resize_img(image_name= "lady.jpg"):
    
    dim = (192,256)
    img = cv2.imread(image_name)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imwrite("data//test//image//"+image_name.split('.')[0]+".jpg" , resized)

def resize_cloth(image_name= "cloth.jpg" , person_name = "lady.jpg"):
    dim = (192,256)
    img = cv2.imread(image_name)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("data//test//cloth//"+person_name.split('.')[0]+".jpg" , resized)


def mask_cloth(image_name= "lady.jpg"):
    loc = "data//test//cloth//"+image_name.split('.')[0]+".jpg" 
    write_loc = "data//test//cloth-mask//"+image_name.split('.')[0]+".jpg" 

    image = cv2.imread(loc)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    rect = (1, 1, mask.shape[1], mask.shape[0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
                                           fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    # save mask
    cv2.imwrite(write_loc, outputMask)

    # display mask
    # cv2.imshow('mask',outputMask)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def mask_img(image_name= "lady.jpg"):
    loc = "data//test//image//"+image_name.split('.')[0]+".jpg" 
    loc1 = "data//test//image//"+image_name.split('.')[0]+".png" 
    write_loc = "data//test//image-mask//"+image_name.split('.')[0]+".png"

    removeBgAPI(loc,loc1)

    image = cv2.imread(loc1)
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    rect = (1, 1, mask.shape[1], mask.shape[0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
                                           fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    
    # save mask
    cv2.imwrite(write_loc, outputMask)

    
    cv2.imshow('mask',outputMask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mask_img_simple(image_name= "lady.jpg"):
    loc = "data//test//image//"+image_name.split('.')[0]+".jpg" 
    #loc1 = "data//test//image//"+image_name.split('.')[0]+".png" 
    write_loc = "data//test//image-mask//"+image_name.split('.')[0]+".png"

    

    image = cv2.imread(loc)
    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    rect = (1, 1, mask.shape[1], mask.shape[0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
                                           fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    
    # save mask
    # cv2.imwrite(write_loc, outputMask)

    
    cv2.imshow('mask',outputMask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mask_img1(image_name= "lady.jpg"):
    loc = "data//test//image//"+image_name.split('.')[0]+".jpg"
    #loc = "data//test//image-parse-new//"+image_name.split('.')[0]+".png" 
    write_loc = "data//test//image-mask//"+image_name.split('.')[0]+".png" 


    getMaskAPI(loc,write_loc)
    
    parse_shape = cv2.imread(write_loc)
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(parse_shape, -1, kernel)

    cv2.imwrite(write_loc , sharpened)
    # cv2.imshow('mask',sharpened)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def pose_estimate(image_name= "lady.jpg"):
    res = None
    IMAGE_FILES = ["data//test//image//"+image_name.split('.')[0]+".jpg"]
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        res = results.pose_landmarks.landmark
    pose_keypoints = {}
    final_pose = {}
    places = [0,10 , 12 , 14 , 16 , 11 , 13,15,24,26,28 , 23,25,27,5,2,8,7]
    count = 0
    for i in range(0,54,3):
        idx = places[count]
        if count==1:
            pose_keypoints[i] = (res[11].x + res[12].x) * 0.5 * image_width
            pose_keypoints[i+1] = res[idx].y * image_height
            pose_keypoints[i+2] = res[idx].visibility
        else:
            pose_keypoints[i] = res[idx].x * image_width
            pose_keypoints[i+1] = res[idx].y * image_height
            pose_keypoints[i+2] = res[idx].visibility
        count +=1
    pose_keypoints_arr = []
    for i in pose_keypoints.values():
        pose_keypoints_arr.append(i)
    final_pose['version'] = 1.0
    final_pose['people'] = [{'face_keypoints':[] ,'pose_keypoints':pose_keypoints_arr,'hand_right_keypoints':[],'hand_left_keypoints':[] }]
    with open("data//test//pose//"+image_name.split('.')[0]+"_keypoints.json", "w") as outfile:
        json.dump(final_pose, outfile)


def parse_img(image_name= "lady.jpg"):
    loc = "./data//test//image//"+image_name.split('.')[0]+".jpg" 
    write_loc = "./data//test//image-parse//"+image_name.split('.')[0]+".png" 
    write_loc_new = "./data//test//image-parse-new//"+image_name.split('.')[0]+".png" 
    
    JPPNet_parsing(loc, 'checkpoints/JPPNet-s2', write_loc,write_loc_new)

    img = cv2.imread("data//test//image-parse//"+image_name.split('.')[0]+".png")

    # cv2.imshow('mask',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    
if __name__ == '__main__':
    resize_img() #pass person img loc
    pose_estimate() 
    resize_cloth() #pass cloth loc
    mask_cloth()
    mask_img()
    parse_img()
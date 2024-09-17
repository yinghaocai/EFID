'''
Data: 2024-5-4
Auther: Yinghao Cai
Description: LC algorithm is one of base algorithms of Image saliency.\
  The sum of the gray value distance of each pixel from all other pixels in the image\
  is used to represent the significant value of that pixel, and calculated using the histogram.
Usage:
python LC.py --input ./Ori/Positive --output LC/Positive
python LC.py --input ./Ori/Negative --output LC/Negative
'''

import os
import cv2
import argparse
import numpy as np
from imutils.paths import list_images

def read_image(image_path):
    image = cv2.imread(image_path)
    # the Narrow Edge of Pictures
    min_edge = min(image.shape[0], image.shape[1])  
    
    # Set the Scaling Proportion
    proportion = 1  
    if min_edge > 3000:
        proportion = 0.1
    elif 2000 < min_edge <= 3000:
        proportion = 0.2
    elif 1000 < min_edge <= 2000:
        proportion = 0.3
    elif 700 <= min_edge <= 1000:
        proportion = 0.4
    
    # Reset Images
    resize_image = cv2.resize(image, None, fx=proportion, fy=proportion, interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    return image_gray

def LC(image_gray):
    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]
    image_gray_copy = np.zeros((image_height, image_width))
    
    # Histogram, counting the number of each grayscale value in the image.
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    
    # Distance of grayscale values from other values
    gray_dist = cal_dist(hist_array)  
    # print(gray_dist)
    for i in range(image_width):
        for j in range(image_height):
            temp = image_gray[j][i]
            image_gray_copy[j][i] = gray_dist[temp]
    image_gray_copy = (image_gray_copy - np.min(image_gray_copy)) / (np.max(image_gray_copy) - np.min(image_gray_copy))
    return image_gray_copy

def cal_dist(hist):
    dist = {}
    for gray in range(256):
        value = 0.0
        for k in range(256):
            value += hist[k][0] * abs(gray - k)
        dist[gray] = value
    return dist

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
      help="path to input dataset")
    ap.add_argument("-o", "--output", required=True,
      help="path to output dataset")
    args = vars(ap.parse_args())

    input_basic = args['input']
    output_basic = args['output']
    if not os.path.exists(output_basic):
        os.makedirs(output_basic)
    for image in list_images(input_basic):
        image_name = os.path.split(image)[-1]
        image_target = os.path.join(output_basic, image_name)
        
        image_gray = read_image(image)
        image_gray_copy = LC(image_gray)
        
        cv2.imwrite(image_target, image_gray_copy * 255 )


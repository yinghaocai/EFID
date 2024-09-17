'''
Data: 2024-5-4
Auther: Yinghao Cai
Description: FT algorithm is one of base algorithms of Image saliency.\
  The main features utilized are color and brightness characteristics.
Usage:
python FT.py --input ./Ori/Positive --output FT/Positive
python FT.py --input ./Ori/Negative --output FT/Negative
'''
from imutils.paths import list_images
from skimage import img_as_ubyte
from imageio import imwrite
import numpy as np
import argparse
import cv2
import os

def FT(path_src, path_target):
  img = cv2.imread(path_src)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.GaussianBlur(img,(5,5), 0)
  gray_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

  l_mean = np.mean(gray_lab[:,:,0])
  a_mean = np.mean(gray_lab[:,:,1])
  b_mean = np.mean(gray_lab[:,:,2])
  lab = np.square(gray_lab- np.array([l_mean, a_mean, b_mean]))
  lab = np.sum(lab,axis=2)
  lab = lab/np.max(lab)
  
  imwrite(path_target, img_as_ubyte(lab), format = 'png')
  
  return None
  
  
if __name__ == '__main__':
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
    img_target = os.path.join(output_basic, image_name)
    FT(image, img_target)


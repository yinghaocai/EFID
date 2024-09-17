# coding=utf-8
# Copyright 2024 Yinghao Cai
#
# Classic Deep Neural Networks for Detecting Paleontology Footprint
# 
# Cite our paper: 

# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import cv2
import os

# def load_data(imageFile):
def load_data(nps):
  in_size = 224
  print("[INFO] loading images...")
  imagePaths = list(paths.list_images(nps["dataset_dir"]))
  data = []
  labels = []
  for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (in_size, in_size))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
    
  # convert the data and labels to NumPy arrays while scaling the pixel
  # intensities to the range [0, 255]
  data = np.array(data) / 255.0
  labels = np.array(labels)
  print('data.shape', data.shape)
  print('labels.shape', labels.shape)
  
  # perform one-hot encoding on the labels
  le = LabelEncoder()
  labels = le.fit_transform(labels)
  num_classes = len(le.classes_)
  labels = to_categorical(labels, num_classes)
  
  # partition the data into training and testing splits using 50% of
  # the data for training， 30% of the data for validation
  # and the remaining 20% for testing
  def train_validation_test_split(data, labels):
    (x_train, non_x_train, y_train, non_y_train) = train_test_split(data, labels, test_size=0.50, stratify=labels, random_state=44)
    (x_val, x_test, y_val, y_test) = train_test_split(non_x_train, non_y_train, test_size=0.40, stratify=non_y_train, random_state=44)# 0.4 == 20% / 50%
    return x_train, x_val, x_test, y_train, y_val, y_test
  
  # adding useful parameters to nps
  nps['num_classes'] = num_classes
  nps['name_classes'] = le.classes_
  nps['x_train'], nps['x_val'], nps['x_test'], nps['y_train'], nps['y_val'], nps['y_test'] = train_validation_test_split(data, labels)

  return nps

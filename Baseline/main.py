# coding=utf-8
# Copyright 2024 Yinghao Cai
#
# Classic Deep Neural Networks for Detecting Paleontology Footprint
# 
# Cite our paper: 

# import the necessary library packages
from __future__ import print_function
from keras.utils import to_categorical
import argparse
import os

# import the user-defined packages
from model_frame import ModelFrame
from data_load import load_data

if __name__ == '__main__':
  # 1. Construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-d", "--dataset", type=str, default="pf",
    help="dataset name, default is Paleontology-Footprint")
  ap.add_argument("-dd", "--dataset_dir", type=str, required=True,
    help="directory containing Paleontology-Footprint-Recognition images")
  ap.add_argument("-mt", "--model_type", type=str, required=True,
    help="directory containing COVID-19 images")
  #ap.add_argument("-nc", "--num_classes", type=int, default=2,
  #  help="number of classes, default is 2, covid19 and normal")
  #ap.add_argument("-p", "--plot", type=str, default="plot.png",
  #  help="path to output loss/accuracy plot")
  #ap.add_argument("-m", "--model", type=str, default="keras_covid19",
  #  help="path to output loss/accuracy plot")
  ap.add_argument("-bs", "--batch_size", type=int, default=8,
    help="batch size, default is 8")
  ap.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
    help="learning rate, default is 1e-3")
  ap.add_argument("-wd", "--weight_decay", type=float, default=0.0005,
    help="weight decay, default is 0.0005")
  ap.add_argument("-me", "--max_epochs", type=int, default=25,
    help="max epoches, default is 25")
  args = vars(ap.parse_args())
  if args['model_type'] not in ["google", "vgg16", "le", "alex", "res34"]:
    print('Error: model type should be in ["google", "vgg16", "le", "alex", "res34"]')
    exit()

  # 2. Load images
  # adding parameters "x_train", "x_test", "y_train", "y_train", 
  # "y_test", "num_classes" and "name_classes" according "dataset_dir" 
  args = load_data(args)
  
  # 3. Create and train the model
  model = ModelFrame(args, True)
    
  # 4. Predict the result
  predicted_x = model.predict()

  # 5. Evaluate the result
  model.evaluate()
  

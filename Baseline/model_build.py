# coding=utf-8
# Copyright 2024 Yinghao Cai
#
# Classic Deep Neural Networks for Detecting Paleontology Footprint
# 
# Cite our paper: 

# import the necessary packages
from __future__ import print_function
from keras.layers import Dropout, Flatten, AveragePooling2D, concatenate
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.layers import Input, Conv2D, Dense, ZeroPadding2D, add
from tensorflow.keras import regularizers
from keras.initializers import he_normal
from keras.models import Model
import keras

def GoogLeNet(x_shape, num_classes):
  def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):  
    if name is not None:  
      bn_name = name + '_bn'  
      conv_name = name + '_conv'  
    else:  
      bn_name = None  
      conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  
  
  def Inception(x,nb_filter):  
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)  
  
    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)  
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)  
  
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)  
  
    return x  
  
  # defining input image size
  inputs = Input(shape=x_shape) # (224,224,3)
  
  # 1st conv layer
  conv1 = Conv2d_BN(inputs,64,(7,7),strides=(2,2),padding='same')  
  pooling1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(conv1)
  
  # 2nd conv layer
  conv2 = Conv2d_BN(pooling1,192,(3,3),strides=(1,1),padding='same')  
  pooling2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(conv2)
  
  # 3rd conv layer
  inception3a = Inception(pooling2,64)#256  
  inception3b = Inception(inception3a,120)#480  
  pooling3 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(inception3b)
  
  # 4th conv layer
  inception4a = Inception(pooling3,128)#512 
  inception4b = Inception(inception4a,128)#512 
  inception4c = Inception(inception4b,128)#512
  inception4d = Inception(inception4c,132)#528  
  inception4e = Inception(inception4d,208)#832  
  pooling4 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(inception4e)
  
  # 5th conv layer
  inception5a = Inception(pooling4,208)#832  
  inception5b = Inception(inception5a,256)#1024  
  pooling5 = AveragePooling2D(pool_size=(7,7),strides=(1,1),padding='same')(inception5b)
  
  # flattening before sending to fully connected layers
  flatten = Flatten()(pooling5) 

  # fully connected layers
  dropout = Dropout(0.4)(flatten)
  dense1 = Dense(num_classes,activation='relu')(dropout)  
  dense1 = BatchNormalization()(dense1)
  dense2 = Dense(num_classes,activation='softmax')(dense1)
  
  model = Model(inputs,dense2,name='inception')
  return model

def AlexNet(x_shape, num_classes):
  # building parameters
  DATA_FORMAT = 'channels_last'

  # defining input image size
  img_input = Input(shape=x_shape)

  # 1st conv layer
  x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
           activation='relu', kernel_initializer='uniform')(img_input)  # valid
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(
      2, 2), padding='same', data_format=DATA_FORMAT)(x)
  
  # 2nd conv layer
  x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding='same', data_format=DATA_FORMAT)(x)
  
  # 3rd conv layer
  x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  
  # 4th conv layer
  x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  
  # 5th conv layer
  x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
           activation='relu', kernel_initializer='uniform')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding='same', data_format=DATA_FORMAT)(x)
  
  # flattening before sending to fully connected layers
  x = Flatten()(x)
  # fully connected layers
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  
  # output layer
  output = Dense(num_classes, activation='softmax')(x)
  model = Model(img_input, output)
  # model.summary()
  return model

def LeNet(x_shape, num_classes):

  # defining input image size
  img_input = Input(shape=x_shape)
  
  # 1st conv layer
  conv1 = Conv2D(filters=6, kernel_size=[5, 5], activation='relu', strides=(1, 1), padding = 'same', use_bias=True)(img_input)
  maxpool1 = MaxPooling2D((2, 2), (2, 2), 'same')(conv1)
  
  # 2nd conv layer
  conv2 = Conv2D(filters=16, kernel_size=[5, 5], activation='relu', strides=(1, 1), padding = 'same', use_bias=True)(maxpool1)
  maxpool2 = MaxPooling2D((2, 2), (2, 2), 'same')(conv2)

  # 3rd conv layer
  conv3 = Conv2D(filters=120, kernel_size=[5, 5], activation='relu', strides=(1, 1), padding = 'same', use_bias=True)(maxpool2)

  # flattening before sending to fully connected layers
  flatten = Flatten()(conv3)
  
  # fully connected layers
  dense1 = Dense(84)(flatten)
  dense1 = BatchNormalization()(dense1)
  dense2 = Dense(num_classes, activation='softmax')(dense1)

  model = Model(img_input, dense2)
  # model.summary()
  return model

def VGGNet16(x_shape, num_classes):

  # Define the input layer
  inputs = keras.Input(shape = x_shape)
  
  # Define the converlutional layer 1
  conv1_1 = keras.layers.Conv2D(64, kernel_size= [3, 3], activation= 'relu', padding= 'same')(inputs)
  conv1_2 = keras.layers.Conv2D(64, kernel_size= [3, 3], activation= 'relu', padding= 'same')(conv1_1)
  pooling1 = keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'same')(conv1_2)

  # Define the converlutional layer 2
  conv2_1 = keras.layers.Conv2D(128, kernel_size= [3, 3], activation= 'relu', padding= 'same')(pooling1)
  conv2_2 = keras.layers.Conv2D(128, kernel_size= [3, 3], activation= 'relu', padding= 'same')(conv2_1)
  pooling2 = keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'same')(conv2_2)

  # Define the converlutional layer 3
  conv3_1 = keras.layers.Conv2D(256, kernel_size= [3, 3], activation= 'relu', padding= 'same')(pooling2)
  conv3_2 = keras.layers.Conv2D(256, kernel_size= [3, 3], activation= 'relu', padding= 'same')(conv3_1)
  conv3_3 = keras.layers.Conv2D(256, kernel_size= [1, 1], activation= 'relu', padding= 'same')(conv3_2)
  # conv3_4 = keras.layers.Conv2D(256, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(conv3_3)
  pooling3 = keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'same')(conv3_3)
  
  # Define the converlutional layer 4
  conv4_1 = keras.layers.Conv2D(512, kernel_size= [3, 3], activation= 'relu', padding= 'same')(pooling3)
  conv4_2 = keras.layers.Conv2D(512, kernel_size= [3, 3], activation= 'relu', padding= 'same')(conv4_1)
  conv4_3 = keras.layers.Conv2D(512, kernel_size= [1, 1], activation= 'relu', padding= 'same')(conv4_2)
  # conv4_4 = keras.layers.Conv2D(512, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(conv4_3)
  pooling4 = keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'same')(conv4_3)
  
  # Define the converlutional layer 5
  conv5_1 = keras.layers.Conv2D(512, kernel_size= [3, 3], activation= 'relu', padding= 'same')(pooling4)
  conv5_2 = keras.layers.Conv2D(512, kernel_size= [3, 3], activation= 'relu', padding= 'same')(conv5_1)
  conv5_3 = keras.layers.Conv2D(512, kernel_size= [1, 1], activation= 'relu', padding= 'same')(conv5_2)
  # conv5_4 = keras.layers.Conv2D(512, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'same')(conv5_3)
  pooling5 = keras.layers.MaxPooling2D(pool_size= [2, 2], strides= [2, 2], padding= 'same')(conv5_3)
  
  flatten = keras.layers.Flatten()(pooling5)

  # Defien the fully connected layer
  fc1 = keras.layers.Dense(4096, activation= 'relu')(flatten)
  fc1 = Dropout(0.5)(fc1)
  fc1 = BatchNormalization()(fc1)
  fc2 = keras.layers.Dense(4096, activation= 'relu')(fc1)
  fc2 = Dropout(0.5)(fc2)
  fc2 = BatchNormalization()(fc2)
  prediction = keras.layers.Dense(num_classes, activation= 'softmax')(fc2)

  model = keras.Model(inputs= inputs, outputs = prediction)
  return model

def ResNet34(x_shape, num_classes):
  def conv_block(inputs, 
          neuron_num, 
          kernel_size,  
          use_bias, 
          padding= 'same',
          strides= (1, 1),
          with_conv_short_cut = False):
      conv1 = Conv2D(
          neuron_num,
          kernel_size = kernel_size,
          activation= 'relu',
          strides= strides,
          use_bias= use_bias,
          padding= padding
      )(inputs)
      conv1 = BatchNormalization()(conv1)

      conv2 = Conv2D(
          neuron_num,
          kernel_size= kernel_size,
          activation= 'relu',
          use_bias= use_bias,
          padding= padding)(conv1)
      conv2 = BatchNormalization()(conv2)

      if with_conv_short_cut:
          inputs = Conv2D(
              neuron_num, 
              kernel_size= kernel_size,
              strides= strides,
              use_bias= use_bias,
              padding= padding
              )(inputs)
          return add([inputs, conv2])
      else:
          return add([inputs, conv2])

  inputs = Input(shape= x_shape)
  # x = ZeroPadding2D((3, 3))(inputs)

  # Define the converlutional block 1
  x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(inputs)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

  # Define the converlutional block 2
  x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

  # Define the converlutional block 3
  x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
  x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

  # Define the converlutional block 4
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

  # Define the converltional block 5
  x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
  x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
  x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
  x = AveragePooling2D(pool_size=(7, 7))(x)
  x = Flatten()(x)
  x = BatchNormalization()(x)
  x = Dense(num_classes, activation='softmax')(x)

  model = Model(inputs= inputs, outputs= x)
  # Print the detail of the model
  # model.summary()
  return model

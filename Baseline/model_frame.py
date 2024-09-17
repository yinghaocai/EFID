# coding=utf-8
# Copyright 2024 Yinghao Cai
#
# Classic Deep Neural Networks for Detecting Paleontology Footprint
# 
# Cite our paper: 

# import the necessary library packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv

# import the user-defined packages
from model_build import GoogLeNet, AlexNet, LeNet, VGGNet16, ResNet34


class ModelFrame:
  def __init__(self, hps, train=True): 
    self.dataset      = hps["dataset"]
    self.num_classes  = hps["num_classes"] #10
    self.name_classes = hps["name_classes"]
    self.learning_rate= hps["learning_rate"] #1e-3
    self.weight_decay = hps["weight_decay"] #0.0005
    self.batch_size   = hps["batch_size"]
    self.max_epochs  = hps["max_epochs"]
    self.x_train = hps["x_train"]
    self.x_val   = hps["x_val"]
    self.x_test  = hps["x_test"]
    self.y_train = hps["y_train"]
    self.y_val   = hps["y_val"]
    self.y_test  = hps["y_test"]
    self.x_shape = self.x_train.shape[1:]
    self.hps = hps # other hp
    self.dataset_name = os.path.split(hps["dataset_dir"])[-1]
    self.model_type = hps["model_type"]
    self.name = "%s_%s_%d" % (self.model_type, self.dataset, self.batch_size)
    
    self.model_dir = os.path.join(os.getcwd(), "%s_model" % self.name)
    self.final_weights_file = "%s_%d.h5" % (self.name, self.max_epochs)
    self.model = self.choose_model()
    if train:
      self.model = self.train(self.model)
    else: # load the saved model
      self.model.load_weights(os.path.join(
        self.model_dir, self.final_weights_file))

  def choose_model(self):
    # from model_build import GoogLeNet, AlexNet, LeNet, VGGNet16, ResNet34
    x_shape = self.x_train.shape[1:]
    num_classes = self.num_classes
    
    if self.model_type == 'google':
      return GoogLeNet(x_shape, num_classes)
    if self.model_type == 'alex':
      return AlexNet(x_shape, num_classes)
    if self.model_type == 'le':
      return LeNet(x_shape, num_classes)
    if self.model_type == 'vgg16':
      return VGGNet16(x_shape, num_classes)
    if self.model_type == 'res34':
      return ResNet34(x_shape, num_classes)

  def predict(self):
    def create_csv(csv_path, csv_head):
      with open(csv_path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

    def write_csv(csv_path, csv_data):
      with open(csv_path, 'a+', newline='') as f:
        csv_write = csv.writer(f, dialect = 'excel')
        csv_write.writerow(csv_data)
    x = self.x_test
    dataset_name = self.dataset_name
    model_type = self.model_type
    y = self.y_test
    predIdxs = self.model.predict(x, self.batch_size)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # show a nicely formatted classification report
    print(classification_report(y.argmax(axis=1), predIdxs, target_names=self.name_classes))
    
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, precision, sensitivity and specificity
    cm = confusion_matrix(y.argmax(axis=1), predIdxs) # TN = cm[0, 0], FP = cm[0, 1], FN = cm[1, 0], TP = cm[1, 1]
    total = sum(sum(cm))
    accuracy =  (cm[0][0] + cm[1][1]) / total
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
    specificity = cm[0][0] / (cm[0][1] + cm[0][0])
	
    # show the confusion matrix, accuracy, precision, sensitivity, and specificity
    print(cm)
    print("accuracy: {:.4f}".format(accuracy))
    print("precision: {:.4f}".format(precision))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    result_path = './results.csv'
    csv_head = ['dataset', 'classifier', 'accuracy', 'precision', 'sensitivity', 'specificity']
    csv_data = [dataset_name, model_type, 
					"{:.2f}%".format(accuracy * 100),  "{:.2f}%".format(precision * 100), 
					"{:.2f}%".format(sensitivity * 100), "{:.2f}%".format(specificity* 100)]
    if not os.path.exists(result_path):
      create_csv(result_path, csv_head)
    write_csv(result_path, csv_data)
	
  def evaluate(self):
    x = self.x_test
    y = self.y_test
    test_loss, test_acc = self.model.evaluate(x, y)
    print("test_loss: {:.4f}".format(test_loss))
    print("test_acc: {:.4f}".format(test_acc))
    return test_loss, test_acc

  def train(self, model):
    #training parameters
    batch_size = self.batch_size # 8
    max_epochs = self.max_epochs # 25
    learning_rate = self.learning_rate # 0.001
    lr_decay = learning_rate / max_epochs #1e-6
    
    # The data, shuffled and split between train and validation sets:
    x_train = self.x_train
    x_val   = self.x_val

    y_train = self.y_train
    y_val   = self.y_val
    
    # data augmentation
    datagen = ImageDataGenerator(
      rotation_range=15,      # randomly rotate images in the range (degrees, 0 to 180)
      fill_mode="nearest")  

    # optimization details
    opt = Adam(lr=learning_rate, decay=lr_decay)
    # opt = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    if self.num_classes == 2:
      model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    else: #self.num_classes == 3
      model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    
    # check the last training steps
    checkpoint_dir = self.model_dir
    if not os.path.exists(checkpoint_dir):
      os.mkdir(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, '%s_{epoch:04d}.h5' % self.name)
    period = 5
    assert self.max_epochs % period == 0, "please set max epochs as n*%d" % period
    checkpoint_cb = ModelCheckpoint( # Save weights, every 5-epochs. 
        filepath, save_weights_only=True, verbose=1, period=period)
    last_epochs = 0
    for epoch in range(self.max_epochs,0,-1*period):
      #print(filepath)
      value = {"epoch": epoch}
      if os.path.isfile(filepath.format(**value)):
        print("Load saved weights from %s" % filepath.format(**value))
        model.load_weights(filepath.format(**value))
        # TODO: load history
        # history_db.history.append(v)
        last_epochs = epoch
        break

    # Continue training
    H = model.fit_generator(
      datagen.flow(x_train, y_train,batch_size=batch_size),
      steps_per_epoch=x_train.shape[0] // batch_size,
      epochs=max_epochs,
      initial_epoch=last_epochs,
      validation_data=(x_val, y_val),
      callbacks=[checkpoint_cb],
      verbose=1)
    
    # Save the final weights
    model.save_weights(os.path.join(self.model_dir, self.final_weights_file))
    if last_epochs<max_epochs: # plot the training loss and accuracy
      # print(H.history)
      N = max_epochs
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
      plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
      if "accuracy" in H.history:
        plt.plot(np.arange(0, N), H.history["val_acc"], label="train_acc")
      else:
        # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
      # plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
      plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
      plt.title("Training Loss and Accuracy (%s)" % self.name)
      plt.xlabel("Epoch #")
      plt.ylabel("Loss/Accuracy")
      plt.legend(loc="lower left")
      plot_path = os.path.join(
        self.model_dir, "%s_plot_%04d.png" % (self.name,self.max_epochs))
      plt.savefig(plot_path)
    else: # 
      print("Train from 1st epoch if needing to plot H.history!")
    return model

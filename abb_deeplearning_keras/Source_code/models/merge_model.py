#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:17:15 2017

@author: maverick
"""


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from PIL import Image
import inputgenerator as ig
import json, sys
from keras.utils.visualize_util import plot
import resnet
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score
import cav_gen
import ms_gen
import itertools




cav_train = cav_gen.training_data_generator
ms_train = ms_gen.training_data_generator
cav_val = cav_gen.validation_data_generator
ms_val = ms_gen.validation_data_generator


def merge_train_input_generators(cav_train,ms_train):
    for cx,cy in itertools.izip(cav_train, ms_train):
#        print (cx[0].shape)
#        print (cy[0].shape)
        X_train = np.concatenate((cx[0],cy[0]),axis=0)
        Y_train = np.concatenate((cx[1],cy[1]),axis=0)
        yield X_train, Y_train


def merge_validation_input_generators(cav_val,ms_val):
    for cx,cy in itertools.izip(cav_val, ms_val):
#        print (cx[0].shape)
#        print (cy[0].shape)
        X_val = np.concatenate((cx[0],cy[0]),axis=0)
        Y_val = np.concatenate((cx[1],cy[1]),axis=0)
        yield X_val, Y_val
        
        
joint_train_input_generator = merge_train_input_generators(cav_train,ms_train)
joint_val_input_generator = merge_validation_input_generators(cav_val,ms_val)


samples_per_epoch = len(cav_gen.files_train) + len(ms_gen.files_train)
validation_samples = len(cav_gen.files_validate) + len(ms_gen.files_validate)
    
print ("cav-train ", len(cav_gen.files_train))
print ("ms-train ", len(ms_gen.files_train))
print ("cav-val ", len(cav_gen.files_validate))
print ("ms-val ", len(ms_gen.files_validate))

                                                                                                                                                                                                                                                                                     
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=30)
csv_logger = CSVLogger('/home/pdinesh/knet-euryale/euryale/checkpoint/JT_resnet18_4class.csv')
 # checkpoint for every improved epoch
#    filepath="checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint only the best epoch
filepath="/home/pdinesh/knet-euryale/euryale/checkpoint/JT_RS18_6chan_4CLASS_retrain_day-day_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
# tensorboard visualization
visualizer = TensorBoard(log_dir='checkpoint/logs', histogram_freq=1, write_graph=True)

resnet_model = resnet.ResnetBuilder.build_resnet_18((6, 224, 224), 4) # (img_rows*sequence_length) should be used, for normal sequence (3 color channels)
resnet_model.compile(loss='categorical_crossentropy',
   optimizer='adam',
   metrics=['accuracy'])


plot(resnet_model, show_shapes=True,to_file="checkpoint/resnet_model_18")   
print (resnet_model.summary())

    #pre-initialized weights    
resnet_model.load_weights("/home/pdinesh/knet-euryale/euryale/checkpoint/exp_44_JT18_C4/JT_RS18_6chan_4CLASS_day-day_E26_0.84_weights.best.hdf5")
resnet_model.fit_generator(joint_train_input_generator, 
                 samples_per_epoch=samples_per_epoch, 
                 nb_epoch = 200,
                 verbose=1, 
                 validation_data=joint_val_input_generator,
                 nb_val_samples=validation_samples,
                 max_q_size=100,
                 callbacks=[lr_reducer, early_stopper, csv_logger, checkpoint])#, visualizer])





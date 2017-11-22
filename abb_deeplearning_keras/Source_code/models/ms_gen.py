#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:31:56 2017

Data generator pipeline for MS

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
import random
random.seed(1337)



#json_file = sys.argv[1]
json_file = '/home/pdinesh/knet-euryale/models/params_ms_euryale.json'
json_file = open(json_file, "r")
#    json_file = open('/home/maverick/knet/models/params.json',"r")
params = json.load(json_file)

batch_size = params['batch_size']
nb_classes = params['nb_classes']
nb_epoch = params['nb_epoch']
data_augmentation = True
img_channels = params['img_channels'] # Images are RGB.
# input image dimensions
img_rows = params['img_rows']
img_cols = params['img_cols']

start_date = params['start_date']
end_date = params['end_date']
slice_by_hour = params['slice_by_hour']
start_hour = params['start_hour']
end_hour = params['end_hour']
test_size = params['test_size']
over_sample = params['over_sample']
label_name = params['label_name']
sequence_mode = params['sequence_mode']
sequence_length = params['sequence_length']
overlap = params['overlap']
balanced = params['balanced']
order = params['order']
joint_train_vol = params['joint_train_vol']
joint_val_vol = params['joint_val_vol']

sky_mask = params['sky_mask']
label_mask = params['label_mask']
meta_file = params['meta_file']
root_dir = params['root_dir']



#initializes sky mask and class label mask to be added to the input images
sky_mask = Image.open(sky_mask)
label_mask = {0 : label_mask+"0.png",
       1 : label_mask+"1.png",
       2 : label_mask+"2.png",
       3 : label_mask+"3.png",
       }
for i in range(4):
    label_mask[i] = Image.open(label_mask[i])

    

#initializes sequence_length to be 1 in single image generation cases
sequence_length = sequence_length if sequence_mode else 1

    

files_train, files_validate, labels_train, labels_validate = ig.download_metadata(
                                start_date = start_date,
                                end_date = end_date, 
                                meta_file = meta_file, 
                                root_dir = root_dir,
                                slice_by_hour = slice_by_hour,
                                start_hour = start_hour,
                                end_hour = end_hour,
                                test_size = test_size,
                                nb_classes = nb_classes,
                                over_sample = over_sample,
                                label_name = label_name,
                                sequence_mode = sequence_mode,
                                sequence_length = sequence_length,
                                overlap = overlap,
                                balanced = balanced)    


pre_filter_train = zip(files_train,labels_train)
post_filter_train = random.sample(pre_filter_train, joint_train_vol)
files_train, labels_train = zip(*post_filter_train)

pre_filter_validate = zip(files_validate,labels_validate)
post_filter_validate = random.sample(pre_filter_validate, joint_val_vol)
files_validate, labels_validate = zip(*post_filter_validate)

  

training_data_generator = ig.generate_training_data(
                                file_list = files_train,
                                labels = labels_train,     
                                img_rows = img_rows,
                                img_cols = img_cols,
                                batch_size = batch_size,
                                order = order,
                                nb_classes = nb_classes,
                                sequence_mode = sequence_mode,
                                sequence_length = sequence_length,
                                overlap = overlap,
                                sky_mask=sky_mask,
                                label_mask=label_mask)

validation_data_generator = ig.generate_testing_data(
                                file_list = files_validate,
                                labels = labels_validate,     
                                img_rows = img_rows,
                                img_cols = img_cols,
                                batch_size = batch_size,
                                order = order,
                                nb_classes = nb_classes,
                                sequence_mode = sequence_mode,
                                sequence_length = sequence_length,
                                overlap = overlap,
                                sky_mask=sky_mask,
                                label_mask=label_mask)


samples_per_epoch = len(files_train)
validation_samples = len(files_validate)


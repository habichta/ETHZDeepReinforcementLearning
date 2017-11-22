#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:31:13 2017

@author: maverick
"""

import keras
from keras.applications import vgg16

#model = vgg16.VGG16()
#
#from quiver_engine.server import launch
#launch(model, input_folder='./img', port=7000)



import resnet

model = resnet.ResnetBuilder.build_resnet_18((3, 224, 224), 4)
model.load_weights("/home/maverick/knet/weights/exp_21_singleimage_CAV_RS18_weights.best.hdf5")

from quiver_engine.server import launch
launch(model, input_folder='./img',temp_folder='./tmp', port=7000)

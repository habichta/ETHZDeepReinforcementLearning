#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:04:21 2017

Enables to slice a part of image dataset and generate numpy arrays from the disk.
- Provides easy subsampling through master_index.h5
- 
@author: maverick
"""
from __future__ import print_function
from PIL import Image
from numpy import array
import tarfile
import os, sys, time
from joblib import Parallel, delayed
import multiprocessing
import pandas
import h5py
from datetime import datetime
import numpy as np
import glob
from sklearn.model_selection import train_test_split



def build_file_list(df, root_dir):
    file_list = []
    for index, row in subset.iterrows():
#        print row['folder']+'/'+row['name']
        file_list.append(root_dir+row['folder']+'/'+row['name'])
    return file_list

    
# simple array generator
def generate_array_of_images(file_list):
    return np.array([np.array(Image.open(fname)) for fname in file_list])

# verbose array generator
def array_generator_verbose(file_list):
    length = float(len(file_list))
    i = 0
    img_arr = []

    for fname in file_list:
        i += 1
        if (i%500 == 0):
            print (str((i/length)*100)[:4]+" %")
        #    Image.open(fname)
        img_arr.append(np.array(Image.open(fname)))    
    return np.array(img_arr)




if __name__== "__main__":
    
#    root_dir = sys.argv[1]
    root_dir = "/home/maverick/Desktop/cavriglia_256_files/"
    print ("Reading contents of " +root_dir)

        
    start_time = time.time()
    
    df = pandas.read_hdf('out/master_index.h5')
    subset = df.ix['2015-08-01':'2015-08-01']
    subset['3_bin'] = subset['3_bin'].fillna(0)
    
    
    #slicing for hours/min
    hour = subset.index.hour
    selector = ((12 <= hour) & (hour <= 13)) | ((13 <= hour) & (hour <= 13))
    subset= subset[selector]
    subset = subset.sort_index()
    
    
    
    print ("building list of images")
    file_list = build_file_list(subset, root_dir)
    print (len(file_list))
    
    X_train = generate_array_of_images(file_list)
#    X_train = array_generator_verbose(file_list)
    Y_train = subset['3_bin'].tolist() #label_list
    
    X_train, X_test = train_test_split(X_train, test_size = 0.2)
    Y_train, Y_test = train_test_split(Y_train, test_size = 0.2)
    
        
    print("--- %s seconds ---" % (time.time() - start_time))
    



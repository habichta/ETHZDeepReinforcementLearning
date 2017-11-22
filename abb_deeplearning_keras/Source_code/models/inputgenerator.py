#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: maverick/Arthur Habicht
"""

from __future__ import print_function
from PIL import Image
from PIL import ImageOps

import pandas

import numpy as np
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy import misc

    
def create_image_sequence(img_paths,  sky_mask, img_rows, img_cols,sequence_length):


    img_sequence = []
    for fname in img_paths:
        img_sequence.append(np.array(process_image(fname, sky_mask, img_rows, img_cols)))
    img_sequence = np.concatenate(img_sequence, axis=2) #stacks images on top of each other, to combine color channels (eg: 2rgb imgs = 1 img with 6 channels)
    img_sequence= np.array(img_sequence)
    img_sequence = img_sequence / 255.0

    #Sequence_length is padded to keep up the array shape,
    #padding should be switched to img_cols in case hstack is used instead of vstack

    return img_sequence


#apply mask and resize
def process_image(fname, sky_mask, img_rows, img_cols):
    img = Image.open(fname)
    sky_mask_inv = ImageOps.invert(sky_mask)
    img.paste(sky_mask, mask=sky_mask_inv)
    img = (img.resize((img_rows,img_cols), Image.ANTIALIAS))
    return img


    
def generate_data(
        set_df,
        master_df,
        root_dir,
        img_rows,
        img_cols,
        batch_size,
        order,
        sequence_length,
        sequence_stride,
        sky_mask,
        labels,
        iterate_once=False):



    relative_seq_indices = np.array([-i for i in range(0,sequence_stride*(sequence_length-1)+1,sequence_stride)])

    #last batch may be smaller than batch_size
    while 1:
        batches = (len(set_df)//batch_size)+1
        start = 0
        print("Batches:", batches)
        for batch in range(1, batches):
            end = batch * batch_size
            batch_input = set_df[start:end] # does not throw exception when end is larger than list length.


            X = []
            X_irr=[]
            Y = []


            #create a sample:
            for ind,_ in batch_input.iterrows(): #sample is a single image with its labels

                ix = np.where(master_df.index==ind)[0]
                samples_ix = ix+relative_seq_indices
                sample_data_df = master_df.ix[samples_ix]

                img_folder = sample_data_df['folder'].values[0]
                img_paths = sample_data_df['name'].values #newest image = 0 , oldest = -1 (label image=0)
                img_full_path = [os.path.join(root_dir,img_folder,p) for p in img_paths]

                irr_data = np.array(sample_data_df['irradiation_hs'].values).astype(np.float32) #newest to oldest

                image_sequence = create_image_sequence(img_full_path,sky_mask,img_rows,img_cols,sequence_length).astype(np.float32) #numpy array

                label_data = sample_data_df[labels].values[0] #last image labels

                Y.append(label_data)
                X_irr.append(irr_data[0]) #label irradiance (of newest image)
                X.append(image_sequence)



                #sanity checks
                #indices = [str(e) for e in list(sample_data_df.index)]
                #assert(len(set(sample_data_df.index.date))==1) #no overlap between dates



                #X.append((image_sequence,irradiation_data)) #TODO: check input to resnet
                #Y.append(label_data)

            X_irr = np.array(X_irr)
            X = np.array(X)
            Y = np.array(Y)



            Y = Y.reshape(Y.shape[0], len(labels))
            Y = Y.astype(np.float32)


            start = end
            yield {"img_input":X,"irr_input":X_irr,"date":str(ind)}, Y #X batch of image sequences, Y batch of irr,mpc,balance  of last image in sequence

        if iterate_once:
            break


        
        
#Method to balance all class labels equally.
def balance_classes(df, label_name, nb_classes, over_sample = 1):
    label_count = []
    frames = []
    for i in range(nb_classes):
#        print (i)
        label_count.append(len(df.loc[df[label_name] == i])) #contains count of all classes

    min_volume = min(label_count) #minimum count
    print ("\nCurrent class distribution - " +str(label_count))
    print ("\nAll classes are balanced to - " +str(min_volume*over_sample))
    
    for i in range(nb_classes):
        class_data = df.loc[df[label_name] == i]
        if len(class_data) < (min_volume * over_sample):
            for each in range(over_sample):
                class_data = class_data.sample(n=min_volume)
                frames.append(class_data)
        else:
            class_data = class_data.sample(n=min_volume * over_sample)
            frames.append(class_data)
        
    df = pandas.concat(frames)
    return df


    

# Method to split training, validation data by picking every 'x' day
def split_training_validation_data(df,sequence_length,sequence_stride):

    with open('train_list.out','r') as tr_list,  open('validation_list.out','r') as val_list, open('test_list.out','r') as test_list:


        cut_off = (sequence_length-1)*sequence_stride # cuts of first elements in sets. this is needed since a sample needs to fetch images in the past


        train_set = df[df.index.normalize().isin(tr_list)].sort_index()
        train_set = train_set.drop(train_set.groupby(train_set.index.date).head(cut_off).index)

        validation_set = df[df.index.normalize().isin(val_list)].sort_index()
        validation_set = validation_set.drop(validation_set.groupby(validation_set.index.date).head(cut_off).index)

        test_set = df[df.index.normalize().isin(test_list)].sort_index()
        test_set = test_set.drop(test_set.groupby(test_set.index.date).head(cut_off).index)

        print("Training Set:", len(train_set.index))
        print("Validation Set:", len(validation_set.index))
        print("Test Set:", len(test_set.index))


    return train_set,validation_set,test_set



def download_metadata(start_date, 
                      end_date,
                      data_file,
                      label_file,
                      sky_mask_file,
                      sequence_length,
                      sequence_stride,
                      nb_classes,
                      over_sample,
                      label_name,
                      balanced):

    print("Reading master_data_file...")
    df_data = pandas.read_hdf(data_file)
    print("Reading_master_label file...")
    df_labels = pandas.read_hdf(label_file)
    df = pandas.concat([df_data,df_labels],axis=1)
    df.sort_index()

    subset_df = df.ix[start_date:end_date].sort_index()


    print("Creating sets....")
    train_data, validation_data, test_data = split_training_validation_data(subset_df,sequence_length,sequence_stride)

    #sanity check sets
    """
    intersect1 = train_data.index.intersection(validation_data.index)
    intersect2 = train_data.index.intersection(test_data.index)
    intersect3 = validation_data.index.intersection(test_data.index)
    print("intersections:", intersect1,intersect2,intersect3)
    """

    if (balanced):
        print("Balancing sets....")
        print("Training Set:")
        train_data = balance_classes(train_data, label_name, nb_classes,over_sample)
        """
        print("Validation Set:")
        validation_data = balance_classes(validation_data, label_name, nb_classes, over_sample)
        print("Test Set:")
        test_data = balance_classes(test_data, label_name, nb_classes, over_sample)
        """

    sky_mask = Image.open(sky_mask_file).convert('L')

    return subset_df,train_data,validation_data,test_data,sky_mask





            





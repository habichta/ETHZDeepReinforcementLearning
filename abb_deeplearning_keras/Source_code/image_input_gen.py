#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:51:49 2017

Reads the image contents of all archives in a given root directory
- extracts images into numpy arrays
- builds a query friendly timestamped index in a dataframe
- pixel data and metadata are accessible in the same source
- stores the contents into a HDF file
@author: maverick
"""

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

def read_images_from_archive(source_dir):
    total = 0
    archives = [item for item in os.listdir(source_dir)]
    images = []
    names = []            
    for item in archives:
        if "tar" in item:
            archive = tarfile.open(os.path.join(source_dir, item))
            archive_items = archive.getmembers()
            total = total + len(archive_items)
            print (len(archive_items))
            print ("Total = "+ str(total))
            archive_images, archive_names = convert_to_array(archive, archive_items)
            images.append(archive_images)
            names.append(archive_names)
    return images, names
            

def convert_to_array(archive, archive_items):
        images = []
        names = []
        dataset = {}
        i =0
        for item in archive_items:
            img = archive.extractfile(item)
            try:
#                print img.name
                read_img = Image.open(img)
                img_arr = array(read_img)
                images.append(img_arr)

                # to clean up the tar folder path residue and only pick the name
                img_name = img.name.split('/')[1] 
                img_name = img_name.split('_Debevec')[0]
                img_name = img_name.split('_exp')[0]
                names.append(img_name)
                
                print img_name
                print i
                i += 1
#                print os.path.join(output_dir+'/'+img.name[2:])
#                processed_img.save(os.path.join(output_dir+'/'+img.name[2:]),"JPEG",quality=90)
            except Exception,e:
                print "Error processing " + str(img)
                pass
            
        return images, names
        
def convert_to_df(images,names):
    df = pandas.DataFrame({'name':names[0], 'timestamp':names[0], 'images':images[0]})
    df['timestamp'] = df['timestamp'].apply(lambda val: datetime.strptime(val,"%Y_%m_%d_%H_%M_%S"))
    df.set_index('timestamp',inplace=True)
    
    return df

    
def convert_to_HDF(df):
    print ".."
    df.to_hdf('out/images.h5','df',mode='w',format='fixed',data_columns=True, compression='gzip')
#    store = pandas.read_hdf('out/images.h5')
#    store.ix['2015-07-15':'2015-08-29']


                

if __name__== "__main__":
    
#    root_dir = sys.argv[1]
#    print ("Reading contents of " +root_dir)
    
    start_time = time.time()
#    root_dir = "/home/maverick/Desktop/temp/input"
    test_dir = "/home/maverick/Desktop/temp/chk"
    images, names = read_images_from_archive(test_dir)
    df = convert_to_df(images, names)
    convert_to_HDF(df)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    
    
#    df = pandas.DataFrame({'name':names[0], 'timestamp':names[0], 'images':images[0]})
#    df['timestamp'] = df['timestamp'].apply(lambda val: datetime.strptime(val,"%Y_%m_%d_%H_%M_%S"))
#    df.set_index('timestamp',inplace=True)
#    
#    df.ix['2015-07-15':'2015-08-30']
#    
#    s = '2015_08_28_08_42_52'
#    datetime.strptime(s,"%Y_%m_%d_%H_%M_%S").time()
#    

    archive = tarfile.open("/home/maverick/Desktop/temp/chk/2015_07_16.tar.gz")
    archive_items = archive.getmembers()
    
    mylist = []
    
    for item in archive_items:
        img = archive.extractfile(item)
        read_img = Image.open(img)
        img_arr = np.array(read_img)
        mylist.append(img_arr)
        
        print item
        img = archive.extractfile(item)
        mylist.append(np.array(Image.open(img)))
#        mylist.append(np.array(Image.open(archive.extractfile(item))))
        

    x = np.array([np.array(Image.open(fname)) for item in archive_items])
    
    
    print (len(archive_items))
    data = []
    img = archive.extractfile(archive_items[1])
    read_img = Image.open(img)
    img_arr = np.array(read_img)
    data.append(img_arr)
    name = "2015_07_16_05_21_04_Debevec.jpeg"
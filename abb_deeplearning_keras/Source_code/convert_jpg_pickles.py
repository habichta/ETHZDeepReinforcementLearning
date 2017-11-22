#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:14:53 2017

Loading compressed JPEG images in to memory, prior to feeding to the array generator

@author: maverick
"""

from PIL import Image
from cStringIO import StringIO
#import glob, os
import sys, time
import pandas
#import pickle
import dill as pickle
import numpy as np



def build_file_list(df, root_dir):
    file_list = []
    for index, row in df.iterrows():
#        print row['folder']+'/'+row['name']
        file_list.append(root_dir+row['folder']+'/'+row['name'])
    return file_list

    
def slice_subset(df, start_date, end_date, slice_by_hour=False, start_hour=5, end_hour=21):
    subset = df.ix[start_date:end_date]
    subset['3_bin'] = subset['3_bin'].fillna(0)
    #slicing for hours/min
    if (slice_by_hour):
        hour = subset.index.hour
#        selector = ((12 <= hour) & (hour <= 13)) | ((13 <= hour) & (hour <= 13))
        selector = ((start_hour <= hour) & (hour <= end_hour))
        subset= subset[selector]

    subset = subset.sort_index()
    return subset    
    

# Method builds a dictionary with {key,value} = file_name_with_path, jpg object stored in StringIO
def build_file_object_dictionary(file_list):
    
    length = float(len(file_list))
    print (str(length)+" files")
    i = 0
    img_dict = {}
    
    for img_file in file_list:
        i += 1
        if (i%500 == 0):
            print (str((i/length)*100)[:4]+" %")
                
        if img_file. endswith(".jpeg"):
            jpgdata = StringIO(open(img_file).read())
            img_dict[img_file] = jpgdata
    
    return img_dict
    

    
def save_pickle(content, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def read_pickle(file_name):
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data
    
    
if __name__ == "__main__":
    
    
    start_time = time.time()
    
    root_dir= "/home/maverick/Desktop/cavriglia_256_files/"
    meta_file = '/home/maverick/knet/out/master_index.h5'
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    out_file_name = sys.argv[3]
    
    print ("reading metadata...")
    df = pandas.read_hdf(meta_file)
#    subset = slice_subset(df,"2015-08-01","2015-08-30",False,9,10)
    subset = slice_subset(df,start_date,end_date,False,9,10)
    
    print ("building list of files...")
    file_list = build_file_list(subset, root_dir)
    
    print ("generating file objects...")
#    img_dict = build_file_object_dictionary(file_list)
    
    
    #saving and reading pickle files
    save_pickle(build_file_object_dictionary(file_list), '/home/maverick/knet/out/'+out_file_name+'.pickle')

    
    
    #Test code
#    start_time = time.time()
   
#    data = read_pickle('out/2015-08-01.pickle')
#    print (data == img_dict)
#    print ("old = "+str(len(img_dict))+",new = "+str(len(data)))
 
#    for item in file_list:
#        Image.open(img_dict[item])
    
    
#    for item in file_list:
#        img = np.array(Image.open(data[item]))
#        
#    img = np.array(Image.open(data[file_list[567]]))
#    Image.fromarray(img)    
    
        
    print("--- %s seconds ---" % (time.time() - start_time))
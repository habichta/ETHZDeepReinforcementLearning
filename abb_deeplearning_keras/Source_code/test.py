#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:00:15 2017

HDf file from images, Testing and Benchmark against various methods
@author: maverick
"""

import h5py
import glob
import os.path
import time
#import holopy as hp
from PIL import Image
import tempfile
import numpy as np

ims = [Image.open(f) for f in glob.glob('/home/maverick/Desktop/temp/chk/2015_08_28/*.jpeg')]
data = np.dstack(ims)

def print_stats(name, tstart, tstop):
    size = os.path.getsize(name+'.hdf5')
    print("{0}: {1:.1f}s, {2}MB".format(name, tstop-tstart, size//1e6))

def write_all_images(name, **options):
    f = h5py.File(name+'.hdf5', 'w')
    f.create_dataset('data', data=data, **options)

def benchmark(name, operation):
    tstart = time.time()
    operation(name)
    tstop = time.time()
    print_stats(name, tstart, tstop)

benchmark('raw', lambda x: write_all_images(x))
benchmark('autochunk', lambda x: write_all_images(x, compression='gzip'))
benchmark('chunk_256x256x100', lambda x: write_all_images(x, compression='gzip', chunks=(16,16,100)))



for f in glob.glob('/home/maverick/Desktop/temp/chk/2015_08_28/*.jpeg'):
    print f
    
    data = np.dstack(df['images'])
    
    

# Loading Images Test
    
from PIL import Image                                                            
import numpy as np                                                                    
import matplotlib.pyplot as plt                                                  
import glob
import pandas


df = pandas.read_hdf('out/master_index.h5')
subset = df.ix['2015-10-16':'2015-10-16']


root_folder = "/home/maverick/Desktop/temp/chk/"


#I have two colour images, each 64 X 64 pixels, in the folder
imageFolderPath = '/home/maverick/Desktop/temp/chk/2015_07_16'
imagePath = glob.iglob(imageFolderPath+'/*.jpeg') 
x = np.array([np.array(Image.open(fname)) for fname in imagePath])
#y = np.array([np.array(Image.open(fname)) for fname in imagePath])

mylist = []

for item in glob.iglob(imageFolderPath+'/*.jpeg'):
    mylist.append(np.array(Image.open(item)))
    print item
    
mat = numpy.array(mylist)


z = np.vstack((x,y))

for fname in imagePath:
    y = np.array(Image.open(fname))
    mylist.append(y)
    

for filename in glob.iglob(imageFolderPath+'/*.jpeg'):
     print('/foobar/%s' % filename)
     im_array = np.array( [np.array(Image.open(filename))] )
     
     
     
im_array = numpy.array( [numpy.array(Image.open(imagePath[i])) for i in range(len(imagePath))] )
print im_array.shape
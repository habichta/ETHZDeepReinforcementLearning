#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:19:21 2017

Script to generate transparent masks out of normal ones.
@author: maverick
"""

from PIL import Image

#Resizes an image and converts all white pixels to be transparent
def make_mask_transparent(img_rows, img_cols,img_file):
    mask = Image.open(img_file).resize((img_rows, img_cols))
    mask = mask.convert('RGBA')
    
    pixdata = mask.load()
    for y in xrange(mask.size[1]):
        for x in xrange(mask.size[0]):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    mask.save("out/"+ img_file.split("/")[-1].split(".")[0] +"_256.png" ,"PNG")
    
    
    
#make_mask_transparent(256,256,"/home/maverick/Desktop/temp/cavriglia_skymask.png")


# Test mask by applying on a sample image

test_file = "/home/maverick/Desktop/temp/1.jpeg"

background = Image.open("/home/maverick/Desktop/temp/1.jpeg")

# this file is the transparent one
layer = Image.open("/home/maverick/knet/out/cavriglia_skymask_256.png")
label = Image.open("/home/maverick/knet/out/class/0.png")

img = background
print layer.mode # RGBA


img.paste(layer, (0, 0), mask=layer)
img.paste(label, (0, 0), mask=label)
img = img.resize((96,96), Image.ANTIALIAS)
img.show()


#img.paste(layer, (0, 0), mask=layer) 
# the transparancy layer will be used as the mask

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:57:38 2016
To check raw file count in tarballs, because ¨ls¨ takes eons
@author: Dinesh

Usage:
command line arguments - {source folder}

"""
from __future__ import print_function
import tarfile
import os, sys



def find_file_contents(source_dir):
    total = 0
    archives = [item for item in os.listdir(source_dir)]
                
    for item in archives:
        if "tar" in item:
            archive = tarfile.open(os.path.join(source_dir, item))
            archive_items = archive.getmembers()
            total = total + len(archive_items)
            print (len(archive_items))
            print ("Total = "+ str(total))
        
        

                

if __name__== "__main__":
    
    root_dir = sys.argv[1]
    
    print ("Reading contents of " +root_dir)
    find_file_contents(root_dir)
    

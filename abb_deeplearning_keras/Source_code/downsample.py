# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:43:10 2016
To preprocess image dataset and resize them
@author: Dinesh

Usage:
command line arguments - {img source folder} {the size you want the image to be 128,256 etc} {destination folder}

   example : 
   python resizer.py /home/ubuntu/images/ 128 /home/ubuntu/new_images

"""


from PIL import Image
import tarfile
import os, sys
from joblib import Parallel, delayed
import multiprocessing


#checks if a particular file is an archive .tar .zip .7z etc
def check_file_type(file_name):
    
    result = file_name.lower().endswith(('.tar', '.zip'))
    return result

    
#checks and creates an output folder if it doesn´t exist    
def create_output_folder(folder):
    try:
        folder = folder.split('.')[0] # eg: to drop .tar from 2015_09_05.tar
        folder_path = os.path.join(output_dir+'/', folder+'/')        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path
    except Exception,e:
        print "Error, creating Output directory etc : ", e
        pass


# Reads the contents of a tar archive without extracting it.
def read_tar_archive(item):
    try:
        archive = tarfile.open(os.path.join(input_dir, item))
        archive_items = archive.getmembers()
        # dictionary returns both values
        return {'archive':archive, 'archive_items':archive_items }
    except Exception,e:
        print "Error, reading the archive file : ", e
        pass


#Method to process images in an archive        
def process_images(archive, archive_items, size):
    for item in archive_items:
        img = archive.extractfile(item)
        try:
#            print img.name
            read_img = Image.open(img)
            processed_img = resize_image(read_img, size)
            # to clean up the tar folder path residue and only pick the name
            img_name = img.name.split('/')[2] 
#            print os.path.join(output_dir+'/'+img.name[2:])
            processed_img.save(os.path.join(output_dir+'/'+img.name[2:]),"JPEG",quality=90)
        except Exception,e:
            print "Error processing " + str(img)
            pass

            
# Method to resize images with PIL            
def resize_image(img, size):

    img = img.resize((int(size),int(size)), Image.ANTIALIAS)
#    img.show()
    return img


def parallel_block(folder):
    if check_file_type(folder):
        print folder
        result_dict = read_tar_archive(folder) #reads the archive content
        archive = result_dict["archive"]
        archive_items = result_dict["archive_items"]
        
        result_folder = create_output_folder(folder) #creates a folder for results
        process_images(archive, archive_items, img_size)
    
    

        
#Method to read the input folder and verify its cotents
def read_root_folder():
    try:
        print "starting...."
        print "Colecting data from %s " % input_dir
        folders = [ item for item in os.listdir( input_dir ) ]
        
        Parallel(n_jobs=6)(delayed(parallel_block)(folder) for folder in folders)

        #regular code block without using parallel method
#        for folder in folders:
#            
#            if check_file_type(folder):
#                result_dict = read_tar_archive(folder) #reads the archive content
#                archive = result_dict["archive"]
#                archive_items = result_dict["archive_items"]
#                
#                result_folder = create_output_folder(folder) #creates a folder for results
#                process_images(archive, archive_items, img_size)
                
                
    except Exception,e:
        print "Error, check Input directory etc : ", e
        sys.exit(1)


if __name__== "__main__":
    
    try:
        input_dir  = str(sys.argv[1].rstrip('/'))  #path to img source folder
        img_size   = str(sys.argv[2])  #The image size (128, 256,etc)
        output_dir  = str(sys.argv[3].rstrip('/')) #output directory
        read_root_folder()
        
    except Exception,e:
        print "Error, check Input directory etc : ", e
        sys.exit(1)
    
    

        
        
        


        

############################


#archive = tarfile.open("/home/maverick/Desktop/temp/resize/2015_02_19.tar")
#archive_items = archive.getmembers()
#
#root_dir = "/home/maverick/Desktop/temp/"
#input_dir = "/home/maverick/Desktop/temp/"
##root_dir = root_dir.rstrip('/')
#
#folders = [item for item in os.listdir(root_dir)]
#
#
#for item in archive_items:
#
#    img = archive.extractfile(item)
#    try:
#        print img.name
#        img = Image.open(img)
#        processed_img = resize_image(img, 256)
#    except Exception, e:
#        print "Error processing " + str(img)
#
#    processed_img.save((root_dir + item.name[2:]), "JPEG", quality=90)



    
#a = archive.extractfile(archive_items[4])
#img = Image.open(a)
#img2 = img.resize(int(256), int(256), Image.ANTIALIAS)
#
#b = resize_image(img, 256)
#img1 = img.resize((int(256),int(256)), Image.ANTIALIAS)
#
#b.save(root_dir+ "test.JPEG", quality=90)
#b.save((root_dir + a.name[2:]), "JPEG", quality=90)
#
#
#list_dir =  os.path.join(root_dir, folders[15] )
#xxx = read_tar_archive(folders[15])

########################
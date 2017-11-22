#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:33:04 2017

Builds a raw index of image timestamps with accessor information
- includes file source information (D)
- normalizes timestamps of raw images with information in sensor logs (D)
- includes image labels (D)

@author: maverick
"""

import tarfile
import os, sys
import pandas
from datetime import datetime
import itertools


def read_images_from_archive(source_dir):
    total = 0
    archives = [item for item in os.listdir(source_dir)]
    folder = []
    name = []
    timestamp = []
    for item in archives:
        if "tar" in item:
            archive = tarfile.open(os.path.join(source_dir, item))
            archive_items = archive.getmembers()
            total = total + len(archive_items)
            print (len(archive_items))
            print ("Total = "+ str(total))
            folder_names, file_names, timestamps = build_file_index(archive, archive_items)
            folder.append(folder_names)
            name.append(file_names)
            timestamp.append(timestamps)
            
            
    folder = list(itertools.chain(*folder))
    name = list(itertools.chain(*name))
    timestamp = list(itertools.chain(*timestamp))
            
    return folder, name, timestamp
        
        

def build_file_index(archive, archive_items):
        folder = []
        name = []
        timestamp = []
        i =0
        for item in archive_items:

            try:
                item_name = str(item).split("'")[1]
                folder_name = item_name.split("/")[0]
                file_name = item_name.split("/")[1]
                time = file_name.split('_Debevec')[0]
                time = time.split('_exp')[0]
                time = time.split('_Mertens')[0]
                
                folder.append(folder_name)
                name.append(file_name)
                timestamp.append(time)
                
#                print time
#                print i
                i += 1

            except Exception,e:
                print "Error processing " + str(item)
                pass
            
        return folder, name, timestamp


        
def convert_to_df(folder,name, timestamp):
    df = pandas.DataFrame({'name':name, 'timestamp':timestamp, 'folder':folder})
    df['timestamp'] = df['timestamp'].map(lambda x: x.rstrip('.jpeg')) #to filter out '.jpeg' that got appended to few timestamps
    df['timestamp'] = df['timestamp'].apply(lambda val: datetime.strptime(val,"%Y_%m_%d_%H_%M_%S"))
    df.set_index('timestamp',inplace=True)
    df = df.sort_index()
    
    return df

    
def convert_to_hdf(df, filename):
    df.to_hdf('out/'+ filename +'.h5','df',mode='w',format='table',data_columns=True, compression='gzip')
#    store = pandas.read_hdf('out/metadata.h5')
#    store.ix['2015-07-15':'2015-08-29']

def read_hdf(hdf_file):
    df = pandas.read_hdf(hdf_file)
    return df

def find_nearest_timestamp(row):
    date = row['dt']
    print date
    i = logs.index.searchsorted(date)
    nearest_timestamp = logs.index[i]
    return nearest_timestamp
    
    
if __name__== "__main__":
    
#    root_dir = sys.argv[1]    
#    print ("Reading contents of " +root_dir)
    
    root_dir = "/home/maverick/Desktop/ms_256_archives"
#    root_dir = "/home/maverick/Desktop/temp/chk"
    folder, name, timestamp = read_images_from_archive(root_dir)
    df = convert_to_df(folder,name,timestamp) 
    
    convert_to_hdf(df, 'MS/metadata')
    
    metadata = read_hdf('out/MS/metadata.h5')
    logs = read_hdf('out/MS/labels_logs.h5')
    
    
    ############Specific to MS data to handle non-aligned timestamps###############
    #range of dates for which data is valid/complete
    metadata = metadata.ix['2015-07-15':'2016-04-21']
    logs = logs.ix['2015-07-15':'2016-04-21']

    # finds nearest timestamp match for each image by finding corresponding index value in logs
    metadata['dt'] = metadata.index
    metadata['nearest_time']= metadata.apply(find_nearest_timestamp, axis=1)
    
    #combine logs and metadata by joining nearest_time in metadata to index of logs
    logs['nearest_time'] = logs.index
    master_data = pandas.merge(metadata, logs, on='nearest_time')
    master_data['dt'] = master_data['dt_x']
    master_data.set_index('dt',inplace=True)
    master_data = master_data.sort_index()
    master_data = master_data.drop(master_data.ix['2015-07-26':'2015-07-26'])
    master_data = master_data.drop(master_data.index.get_duplicates())
    convert_to_hdf(master_data,'MS/master_index')
    ########################################
    
    #OLD code to merge logs and metadata on plain 'dt' without finding nearest timestamps
    metadata['dt'] = metadata.index
    logs['dt'] = logs.index
    master_data = pandas.merge(metadata, logs, on='dt')
    master_data.set_index('dt',inplace=True)
    master_data = master_data.sort_index()
#    master_data['3_bin'].plot()
    convert_to_hdf(master_data,'MS/master_index')


    

                            
                            
                            

#    mdata = metadata.ix['2015-07-15':'2015-07-15']
#    ldata = logs.ix['2015-07-15':'2015-07-15']
#    mdata['dt'] = mdata.index
#    ldata['dt'] = ldata.index
#    xdata = pandas.merge(mdata, ldata, on='dt')
#    xdata.set_index('dt',inplace=True)
#    xdata = xdata.sort_index()
#    xdata['3_bin'].plot()
    
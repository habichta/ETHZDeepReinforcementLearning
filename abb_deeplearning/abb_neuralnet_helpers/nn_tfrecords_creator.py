import tensorflow as tf
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as rp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import datetime as dt
import pandas as pd
import warnings



'''
Helper functions for tfrecords creation
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




def _create_balanced_dataset_info(changes,nr_of_samples):

    ratio_changes = float(changes)/float(nr_of_samples)
    balanced_df = pd.DataFrame(data=[(changes,nr_of_samples,ratio_changes)], columns=['changes','len','ratio'])

    return balanced_df




def create_path_label_tfrecords_options(suffix="",solar_station=ac.ABB_Solarstation.C, dates=None, time_ranges=None, img_nr=1,
                                        stride=1, file_filter={"Debevec", ".jpeg"}, automatic_daytime=False, compress=False, show_reconstruction=False,
                                        shuffle=False):
    """
    Creates TFRecord binary files. They are the recommended format for Tensorflow stream inputs (only  non random access!). It saves the image path (not the image
    itself) together with the image's labels saved in the -labes.csv file (see abb_clouddrl_transformation_pipeline for
    functions to create labels). As long as the labels (apart from the path) are floats, this function automatically infers
    the labels. No changes necessary. Also calculates the corresponding weights for weighted loss in training phase
    The tfrecords saves the records in ordered fashion (chronologically for each day) This is important for the reinforcement learning part, where a day should be traversed in chronological order
    can be changed with shuffle option... 
    :param solar_station: Choose C or MS, C Default
    :param dates: Tuple of dates to be considered (list of tuples: [(from.to),..]
    :param time_ranges: Time ranges within each day (list of tuples (from,to), only these images are written to the TFrecrods file
    :param automatic_daytime: Infers the sunrise and sunset times automatically, as defined in abb_constants, overrides time_ranges
    :param show_reconstruction: Print out the reconstructed data from the tfrecords file for each day. This is for debugging purposes only
    :param shuffle: If True, ordering of input is not preserved. Note that if no other options have been set in the abb_read_pipeline, tha files are ordered chronologically for each day
    :param compress: If True, tfrecords file will be compressed. Can shrink size from 3MB to 600KB per day
    :param img_nr: nr of images per sample
    :param stride: steps to next image in the same group f.e.  1 2 3 4 5  with img_nr = 2 and stride = 2 =>  [1 3] [2 4] [3 5]
    :param suffix: Add a suffix to the file name
    :return: Prints a .tfrecords file in the image folder. Size is approx 3MB-10MB per day depending on the labels. The filename contains the suffix and I%S%  with % standing for
    image number and stride per sample. The labels are always those of the last sample
    """

    image_files = rp.read_cld_img_time_range_paths(solar_station=solar_station,
                                                   img_d_tup_l=dates, img_t_tup_l=time_ranges,
                                                   automatic_daytime=automatic_daytime,
                                                   file_filter=file_filter)  # set to automatic_daytime

    for day_files in image_files:  # image_files is a generator that returns the image information for each day

        day_path = list(day_files[0].values())[0].rsplit('/', 1)[0]  # z.b. /media/data/Daten/img_C/C-2016-03-17/

        filename = os.path.join(day_path, day_path.rsplit('/', 1)[1] + '-paths'+'I'+str(img_nr)+'S'+str(stride)+str(suffix))
        filename_tf = filename+'.tfrecords'  # z.b. /media/data/Daten/img_C/C-2016-03-14/C-2016-03-14-pathsI2S0.tfrecords

        filename_balanced = filename+'.balanced' #dataframe containing data for training data balancing


        print("Gathering Data for " + filename_tf)
        image_file_list = list(day_files[
                                   0].values())  # z.b. ['/media/data/Daten/img_C/C-2016-03-17/2016_03_17_16_00_00_Debevec.jpeg', '/media/data/Daten/img_C/C-2016-03-17/2016_03_17_16_00_07_Debevec.jpeg',


        #create list of images, using image_nr and strides

        img_nr = int(img_nr)
        stride = int(stride)



        print("Creating list")
        image_file_list2 = [image_file_list[i:(i+(img_nr*stride)):stride] for i in range(len(image_file_list)-(img_nr-1)*stride)]
        # z.b. [['/media/data/Daten/img_C/C-2015-07-16/2015_07_16_07_14_02_Resize256.jpeg', '/media/data/Daten/img_C/C-2015-07-16/2015_07_16_07_14_41_Resize256.jpeg'], ....


        if len(image_file_list2) == 0:
            raise ValueError('The image list is empty, maybe bad input for img_nr and stride. Or too short time period (too few pictures)')



        
       

        # Read the pandas label dataframe for this image
        label_path = os.path.join(day_path, day_path.rsplit('/', 1)[1] + '-labels.csv')
        day_labels = pd.DataFrame.from_csv(path=label_path)

        # writer for one day, each day gets its TFRecords file

        if compress:
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        else:
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

        writer = tf.python_io.TFRecordWriter(filename_tf, options=options)


        changing_sample_sum = 0
        total_samples = len(image_file_list2)



        for sample in image_file_list2:  # go through all time stamps of the images and create TFrecords with path to file and labels, sample contains of collection of images


            i = 0
            data_dict={}
            for image_path in sample:

                data_dict['image_path'+str(i)] = _bytes_feature(image_path.encode()) #path to each image in a sample image_path1, image_path2, ...

                if i == len(sample)-1: #last image labels are added, if iteration is at last image of sample, here its  labels in the labels dataframe are added as well

                    file_name_parts = image_path.rsplit('/', 1)[1].split('_')[:-1]  # z.b. 2015_07_19_21_17_54
                    pd_key_str = ' '.join(('-'.join(file_name_parts[0:3]), ':'.join(file_name_parts[3:6])))
                    pd_key_dt = dt.datetime.strptime(pd_key_str, '%Y-%m-%d %H:%M:%S')

                    data_dict_temp = day_labels.loc[pd_key_dt].to_dict()

                    #This calculates the statistics of each sample concerning whether it is "changing" or not. changing means whether the B flag in the label dataframe is  1.0 or 0.0
                    changing_info = data_dict_temp['B']
                    changing_sample_sum+=changing_info

                    data_dict_l = {k: _float_feature(v) for (k, v) in data_dict_temp.items()}
                    data_dict.update(data_dict_l)

                i += 1

            example = tf.train.Example(features=tf.train.Features(feature=data_dict))
            writer.write(example.SerializeToString())

        writer.close()

        #Calculate this day's overall statistics concerning the change of weather from cloudy to sunny or sunny to cloudy. Needed for data balancing in training phase
        balance_set_df = _create_balanced_dataset_info(changing_sample_sum,total_samples)
        print(balance_set_df)
        print("Write: ", filename_balanced)
        balance_set_df.to_csv(filename_balanced, sep=',')




        if show_reconstruction:  # check whether written correctly (for debugging), show only path ...,
            record_iter = tf.python_io.tf_record_iterator(path=filename_tf)

            for record in record_iter:
                example = tf.train.Example()
                example.ParseFromString(record)
                path_strings=list()

                for i in range(img_nr):

                    path_strings.append(example.features.feature['image_path'+str(i)].bytes_list.value[0])


                print(path_strings)











def create_path_label_tfrecords(suffix = "", solar_station = ac.ABB_Solarstation.C, file_filter={"Debevec", ".jpeg"}, dates = None, time_ranges = None, automatic_daytime=False, compress = False, show_reconstruction=False, shuffle=False):
    """
    Creates TFRecord binary files. They are the recommended format for Tensorflow stream inputs (only  non random access!). It saves the image path (not the image
    itself) together with the image's labels saved in the -labes.csv file (see abb_clouddrl_transformation_pipeline for
    functions to create labels). As long as the labels (apart from the path) are floats, this function automatically infers
    the labels. No changes necessary.
    The tfrecords saves the records in ordered fashion (chronologically for each day) This is important for the reinforcement learning part, where a day should be traversed in chronological order
    can be changed with shuffle option... 
    :param solar_station: Choose C or MS, C Default
    :param dates: Tuple of dates to be considered (list of tuples: [(from.to),..]
    :param time_ranges: Time ranges within each day (list of tuples (from,to), only these images are written to the TFrecrods file
    :param automatic_daytime: Infers the sunrise and sunset times automatically, as defined in abb_constants, overrides time_ranges
    :param show_reconstruction: Print out the reconstructed data from the tfrecords file for each day. This is for debugging purposes only
    :param shuffle: If True, ordering of input is not preserved. Note that if no other options have been set in the abb_read_pipeline, tha files are ordered chronologically for each day
    :param compress: If True, tfrecords file will be compressed. Can shrink size from 3MB to 600KB per day
    :return: Prints a .tfrecords file in the image folder. Size is approx 3MB per day depending on the labels 
    """

    warnings.warn("deprecated",DeprecationWarning)

    image_files = rp.read_cld_img_time_range_paths(solar_station=solar_station,
        img_d_tup_l=dates, img_t_tup_l=time_ranges,automatic_daytime=automatic_daytime,file_filter=file_filter) #set to automatic_daytime




    for day_files in image_files: #image_files is a generator that returns the image information for each day

        day_path = list(day_files[0].values())[0].rsplit('/',1)[0] # z.b. /media/data/Daten/img_C/C-2016-03-17/
        filename = os.path.join(day_path,day_path.rsplit('/',1)[1]+'-paths'+suffix+'.tfrecords') # z.b. /media/data/Daten/img_C/C-2016-03-14/C-2016-03-14.tfrecords
        image_file_list = list(day_files[0].values()) #z.b. ['/media/data/Daten/img_C/C-2016-03-17/2016_03_17_16_00_00_Debevec.jpeg', '/media/data/Daten/img_C/C-2016-03-17/2016_03_17_16_00_07_Debevec.jpeg',

        #Read the pandas label dataframe for this image
        label_path = os.path.join(day_path,day_path.rsplit('/',1)[1]+'-labels.csv')
        day_labels = pd.DataFrame.from_csv(path=label_path)

        #writer for one day, each day gets its TFRecords file

        if compress:
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        else:
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)


        writer = tf.python_io.TFRecordWriter(filename, options=options)

        #Tensorflow execution graph for this day
        filename_queue = tf.train.string_input_producer(image_file_list,shuffle=shuffle)  # keep order of input! Important for later (reinforcement learning)
        image_reader = tf.WholeFileReader()
        k, image = image_reader.read(filename_queue)

        #TODO: Not necessary to do like this... inefficient
        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)


            for i in range(len(image_file_list)): #go through all time stamps of the images and create TFrecords with path to file and labels
                #print(k.eval())
                #image_tensor = image.eval()

                key= sess.run(k) #image tensor height,width,channel


                #fetch the labels
                #create key to pandas dataframe:
                key_str = str(key)
                #print('Image path:', key_str)
                file_name_parts = key_str.rsplit('/',1)[1].split('_')[:-1] #z.b. 2015_07_19_21_17_54
                pd_key_str = ' '.join(('-'.join(file_name_parts[0:3]),':'.join(file_name_parts[3:6])))
                pd_key_dt = dt.datetime.strptime(pd_key_str, '%Y-%m-%d %H:%M:%S')
                #print('Label:',pd_key_dt)
                data_dict_temp = day_labels.loc[pd_key_dt].to_dict()

                data_dict = { k:_float_feature(v) for (k,v) in data_dict_temp.items()}

                data_dict['image_path'] = _bytes_feature(key)

                example = tf.train.Example(features = tf.train.Features(feature=data_dict))

                writer.write(example.SerializeToString())


            coord.request_stop()
            coord.join(threads)
            writer.close()

            if show_reconstruction:  # check whether written correctly (for debugging), show only path ..., check for duplicates
                record_iter = tf.python_io.tf_record_iterator(path=filename)
                dupl_test = set()
                for record in record_iter:
                    example = tf.train.Example()
                    example.ParseFromString(record)
                    #print(list(example.features.feature.keys()))
                    path_string = example.features.feature['image_path'].bytes_list.value[0]
                    if path_string in dupl_test:
                        print('Duplicate detected: ', path_string)
                    else:
                        dupl_test.add(path_string)
                    print(path_string)


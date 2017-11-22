from abb_deeplearning.abb_neuralnet_helpers import nn_tfrecords_creator
import datetime as dt
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as ac


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("suffix", help="")
parser.add_argument("image_per_sample", help="")
parser.add_argument("strides", help="")
parser.add_argument("image_name", help="")
args = parser.parse_args()


#For all days in C
d_from = ac.c_min_date
d_to = ac.c_max_date


#time range example:
t_from = dt.datetime.strptime('9:00:10', '%H:%M:%S')
t_to = dt.datetime.strptime('12:00:10', '%H:%M:%S')

t_from2 = dt.datetime.strptime('12:00:00', '%H:%M:%S')
t_to2 = dt.datetime.strptime('13:00:10', '%H:%M:%S')

#Note: automatic daytime overrides the time ranges! Creates ordered files, compressed = False!
#nn_tfrecords_creator.create_path_label_tfrecords(dates=[(d_from,d_to)], time_ranges=[(t_from,t_to),(t_from2,t_to2)], show_reconstruction=True,automatic_daytime=True)
file_filter={str(args.image_name), ".jpeg"} #select which images to use  Resize256
suffix = int(args.suffix) #256
image_per_sample = int(args.image_per_sample) #2
strides= int(args.strides) #6

nn_tfrecords_creator.create_path_label_tfrecords_options(suffix=suffix, dates=[(d_from, d_to)], time_ranges=None, img_nr=image_per_sample, stride=strides, show_reconstruction=True, automatic_daytime=True, file_filter=file_filter)

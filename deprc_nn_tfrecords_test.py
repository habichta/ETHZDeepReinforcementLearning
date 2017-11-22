from abb_deeplearning.abb_neuralnet_helpers import nn_tfrecords_creator
import datetime as dt
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as ac


#For all days in C
d_from = ac.c_min_date
d_to = ac.c_max_date

d_from = dt.datetime.strptime('2015-08-10', '%Y-%m-%d')
d_to = dt.datetime.strptime('2015-08-10', '%Y-%m-%d')


#time range example:
t_from = dt.datetime.strptime('9:00:10', '%H:%M:%S')
t_to = dt.datetime.strptime('12:00:10', '%H:%M:%S')

t_from2 = dt.datetime.strptime('12:00:00', '%H:%M:%S')
t_to2 = dt.datetime.strptime('13:00:10', '%H:%M:%S')

#Note: automatic daytime overrides the time ranges! Creates ordered files, compressed = False!
#nn_tfrecords_creator.create_path_label_tfrecords(dates=[(d_from,d_to)], time_ranges=[(t_from,t_to),(t_from2,t_to2)], show_reconstruction=True,automatic_daytime=True)

suffix = 384
nn_tfrecords_creator.create_path_label_tfrecords_options(suffix=suffix, dates=[(d_from, d_to)], time_ranges=[(t_from, t_to)], img_nr=2, stride=1, show_reconstruction=True, automatic_daytime=False, file_filter={"Resize384", ".jpeg"})

import abb_deeplearning.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb_tp
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as abb_c
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abb_r
import os
import pandas as pd
import datetime as dt

d_from = dt.datetime.strptime('2015-07-16', '%Y-%m-%d')
d_to = dt.datetime.strptime('2016-04-25', '%Y-%m-%d')

#t_from = dt.datetime.strptime('07:00:00', '%H:%M:%S')
#t_to = dt.datetime.strptime('21:00:00', '%H:%M:%S')


abb_tp.abb_neural_network_labels_generator(solar_station=abb_c.ABB_Solarstation.C,img_d_tup_l=[(d_from,d_to)],automatic_daytime=True,print_to_csv=True,irr_mpc_value_freq_s=20)


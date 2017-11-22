import abb_deeplearning.abb_data_pipeline.abb_clouddrl_transformation_pipeline as at
import datetime as dt


c_min_date = dt.datetime.strptime('2015-07-16', '%Y-%m-%d') #2015-07-15 is corrupted
c_max_date = dt.datetime.strptime('2016-04-25', '%Y-%m-%d')


d_from = dt.datetime.strptime("2016-03-10", '%Y-%m-%d')
d_to = dt.datetime.strptime("2016-04-25", '%Y-%m-%d')

t_from = dt.datetime.strptime("10:00:00", '%H:%M:%S')
t_to = dt.datetime.strptime("12:00:00", '%H:%M:%S')

at.images_draw_sunspots(scale=256,img_d_tup_l=[(d_from,d_to)],img_t_tup_l=None,automatic_daytime=True)
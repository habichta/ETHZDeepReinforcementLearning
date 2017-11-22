import abb_deeplearning.abb_data_pipeline.abb_clouddrl_clearsky as abb_cs
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as ac
import datetime as dt

d_from = dt.datetime.strptime('2015-10-10', '%Y-%m-%d')

abb_cs.ineichen_series(path_c=ac.c_int_data_path, print_to_csv=True)
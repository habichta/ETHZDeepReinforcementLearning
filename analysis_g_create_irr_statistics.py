

import abb_deeplearning

import abb_deeplearning.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abbr
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants import ABB_Solarstation
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
import datetime as dt

[(ac.c_min_date,ac.c_max_date)]
img_path_dict = {ABB_Solarstation.C: ac.c_img_path,
                 ABB_Solarstation.MS: ac.ms_img_path}

data_path_dict = {ABB_Solarstation.C: ac.c_int_data_path,
                  ABB_Solarstation.MS: ac.ms_int_data_path}




d_from = dt.datetime.strptime('2015-07-19', '%Y-%m-%d')
d_to = dt.datetime.strptime('2015-07-23', '%Y-%m-%d')


abb.abb_create_irradiance_statistics(path=data_path_dict[ABB_Solarstation.C],img_d_tup_l=None ,output_path='/media/data/Daten/data_C_int/C-irradiance-statistics.csv',abb_solarstation=ABB_Solarstation.C, print_to_csv=True)


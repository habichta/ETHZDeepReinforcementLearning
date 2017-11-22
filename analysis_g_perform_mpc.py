
import sys

# sys.path.append('/home/nox/Drive/Documents/Masterarbeit/shared/dlabb')


import abb_deeplearning

#import abb_dlmodule.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb
#from abb_dlmodule.abb_data_pipeline.abb_clouddrl_constants import ABB_Solarstation as abb_st
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abbr
import datetime as dt
import os


path = "/media/data/Daten/data_C"


# abb.abb_linear_interpolate_illuminance(path=path,output_path='/media/data/Daten/data_C_int',abb_solarstation=abb_st.C,error_log=True)


d_from = dt.datetime.strptime('2016-04-05', '%Y-%m-%d')
d_to = dt.datetime.strptime('2016-04-25', '%Y-%m-%d')


"""
for (img_dict, data_path) in abbr.read_cld_img_time_range_paths(img_d_tup_l=[(d_from,d_to)], img_t_tup_l=[(t_from,t_to)]):
	#pass
	print(img_dict,data_path)
"""


from abb_deeplearning.abb_mpc_controller import abb_mpc


day_list = [t[1] for t in abbr.read_cld_img_day_range_paths(img_d_tup_l=[
                                                            (d_from, d_to)])]


#day_list_f = [day for day in day_list if not os.path.isfile(day.rsplit('-',1)[0]+"-mpc100.csv") ]

print("Calculating "+str(len(day_list))+ " days")



abb_mpc.perform_default_mpc(
    day_path_list=day_list, resolution_s=1, time_range=None,interpolate_to_s=True,write_file=True,change_constraint_wh_min=40)

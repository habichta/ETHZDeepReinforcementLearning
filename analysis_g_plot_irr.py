
import argparse
import abb_deeplearning

#import abb_dlmodule.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb
#from abb_dlmodule.abb_data_pipeline.abb_clouddrl_constants import ABB_Solarstation as abb_st
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abbr
import datetime as dt



parser = argparse.ArgumentParser()
parser.add_argument("date", help="date to show")
args = parser.parse_args()



d_from = dt.datetime.strptime(str(args.date), '%Y-%m-%d')
d_to = dt.datetime.strptime(str(args.date), '%Y-%m-%d')


from abb_deeplearning.abb_data_pipeline import abb_clouddrl_visualization_pipeline as abb_vis


day_list = [t[1] for t in abbr.read_cld_img_day_range_paths(img_d_tup_l=[
                                                            (d_from, d_to)])]

print(day_list)
abb_vis.plot_irradiance_mpc(date_path=day_list[0])



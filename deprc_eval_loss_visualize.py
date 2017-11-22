
import argparse
import abb_deeplearning

#import abb_dlmodule.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb
#from abb_dlmodule.abb_data_pipeline.abb_clouddrl_constants import ABB_Solarstation as abb_st
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abbr
import datetime as dt

from abb_deeplearning.abb_data_pipeline import abb_clouddrl_visualization_pipeline as abb_vis

"""
not fully functional
"""

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to dataframe")
parser.add_argument("--date", help="date to show")
args = parser.parse_args()
path = args.path
img_tup = None
if(args.date):
    d_from = dt.datetime.strptime(str(args.date), '%Y-%m-%d')
    d_to = dt.datetime.strptime(str(args.date), '%Y-%m-%d')
    img_tup = [(d_from, d_to)]
    abb_vis.plot_loss_predictions_of_day(date=img_tup[0][0], path=path)
else:
    abb_vis.plot_loss_predictions_of_day(date=None, path=path)







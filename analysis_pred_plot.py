import argparse
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abbr
import datetime as dt


"""
z.b:
python analysis_pred_plot.py /home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT3_EVAL/simple_2_False_regression_simple_dqnnet_do_kp0.5_32_diffTrue_ch3 "2015-08-17 15:01:00"
"""

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to eval prediction file")
parser.add_argument("date", help="date to show")
args = parser.parse_args()

path = args.path




from abb_deeplearning.abb_data_pipeline import abb_clouddrl_visualization_pipeline as abb_vis



abb_vis.plot_predictions_of_day(pred_path=path, pred_nr=10,date=args.date)
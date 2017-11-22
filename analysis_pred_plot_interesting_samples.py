

from abb_deeplearning.abb_data_pipeline import abb_clouddrl_visualization_pipeline as abb_vis
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to eval prediction file")
args = parser.parse_args()

path = args.path



with open("interesting_test_samples.out") as f:
    samples = [l.rstrip() for l in f.readlines()]



abb_vis.save_plot_predictions_of_day(pred_path=path,pred_nr=10,dates=samples)
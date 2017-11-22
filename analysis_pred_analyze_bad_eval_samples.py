
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_visualization_pipeline as abb_vis
import argparse
"""
Analyze where evaluation of model performed particularly bad, defined by outlier settings in prediction
"""
parser = argparse.ArgumentParser()
parser.add_argument('pred_path', nargs='+')

pred_path_l = list()
for _, value in parser.parse_args()._get_kwargs():
    if value is not None:
        pred_path_l.append(value)

print(pred_path_l)
abb_vis.analyze_bad_samples(pred_path=pred_path_l[0])
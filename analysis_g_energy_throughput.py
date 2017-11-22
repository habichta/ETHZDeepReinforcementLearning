import abb_deeplearning.abb_data_pipeline.abb_clouddrl_transformation_pipeline as abb_tp
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as abb_c
import datetime as dt

"""
Calculate difference between Irr and MPC to determine energy throughput of battery

"""



abb_tp.energy_throughput_per_day_optimal(print_to_csv=False,visualize=True)
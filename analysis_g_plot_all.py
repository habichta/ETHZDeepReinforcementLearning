from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import matplotlib.pyplot as plt
import pandas as pd
import os

import datetime as dt

d_from = dt.datetime.strptime('2015-09-28', '%Y-%m-%d')
d_to = dt.datetime.strptime('2015-09-28', '%Y-%m-%d')

day_list = abb_rp.read_cld_img_day_range_paths(suffix='all',img_d_tup_l=[(d_from,d_to)])




for day in day_list:

    df = pd.DataFrame.from_csv(day[1])

    df[['mpc100','int']].plot()
    plt.show()











print(day_list)
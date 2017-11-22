from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt

"Find time differences between samples"

day_list = abb_rp.read_cld_img_day_range_paths()

print(day_list)



path = ac.c_int_data_path
rf_path = os.path.join(path,'rl_data.csv')
df = pd.read_csv(rf_path,index_col=0,parse_dates=True)

delta_list = list()

for day in day_list:
    d = day[0].rsplit('C-')[1]

    df_d = df.loc[d]


    df_d['tvalue'] = df_d.index
    df_d['delta'] = ((df_d['tvalue'] - df_d['tvalue'].shift()).fillna(0)).dt.total_seconds()
    print(d)

    print(df_d['delta'].max(),df_d['delta'].idxmax())

    delta_list.append(df_d['delta'])


delta_df = pd.concat(delta_list,axis=0).to_frame()
delta_df.columns=["delta"]

print("MEDIAN",delta_df.median())

print(delta_df)

delta_df.hist(column='delta',bins=60)

plt.show()
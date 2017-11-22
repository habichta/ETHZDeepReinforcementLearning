from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import matplotlib as plt
import pandas as pd
import os

day_list = abb_rp.read_cld_img_day_range_paths(suffix='all')



data_list = list()
index_list = list()
col_list = ['mpc100','mpc100_new','mpc40','naive100','naive100_new','naive40']

for day in day_list:

    df = pd.DataFrame.from_csv(day[1])

    mpc100_diff = (df['int'] - df['mpc100']).abs().sum()
    mpc100_new_diff = (df['int'] - df['mpc100_new']).abs().sum()
    mpc40_diff = (df['int'] - df['mpc40']).abs().sum()
    naive100_diff = (df['int'] - df['naive100']).abs().sum()
    naive100_new_diff = (df['int'] - df['naive100_new']).abs().sum()
    naive40_diff = (df['int'] - df['naive40']).abs().sum()

    index = df.index[0].normalize()

    data_list.append([mpc100_diff,mpc100_new_diff,mpc40_diff,naive100_diff,naive100_new_diff,naive40_diff])

    index_list.append(index)



diff_df = pd.DataFrame(data=data_list,index=index_list,columns=col_list)

output_path = os.path.join(abb_c.c_int_data_path,"C-throughput.csv")
pd.DataFrame.to_csv(diff_df,output_path)

sum_diff = diff_df.sum()

print(diff_df)

print(sum_diff)










print(day_list)
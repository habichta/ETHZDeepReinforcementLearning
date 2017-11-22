from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import matplotlib as plt
import pandas as pd
import os
import argparse
import numpy as np

from abb_deeplearning.abb_mpc_controller import abb_mpc
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline import read_full_int_irr_data
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_ac
import matplotlib.pyplot as plt
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("pred_path", help="date to show")
parser.add_argument("loss_t", help="date to show") #11
args = parser.parse_args()


pred_path = args.pred_path
loss_threshhold = float(args.loss_t)

#reference data in the in data path TODO: change for MS
data_path=abb_ac.c_int_data_path
prefix="C-"
suffix="-all.csv"

#OUTPUT path of testing of RL
full_path = pred_path
full_pred_path = os.path.join(pred_path,"eval_predictions.csv")

df_pred_full = pd.read_csv(full_pred_path,index_col=0,parse_dates=True)

data_list = list()
index_list = list()
col_list = ['rl','mpc100','mpc40','naive100','naive40']

stat_data = []

day_list = set(df_pred_full.index.map(pd.Timestamp.date))
day_list = [str(day) for day in day_list]


print(day_list)

for day in day_list:

    print(day)
    df_pred = df_pred_full.loc[day]

    #Get file with optimal mpc etc..
    date = str(df_pred.index[0].normalize()).split(" ")[0] # get current date
    data_file_name = prefix+date+suffix
    data_file_path = os.path.join(data_path,data_file_name)

    df_optimal = pd.DataFrame.from_csv(data_file_path)

    df_optimal = df_optimal.loc[df_pred.index]

    #print(df_optimal)
    #print(df_pred)

    mpc_pred_diff_t = (df_optimal['int'] - df_pred['ci']).abs()


    print(mpc_pred_diff_t.values)
    stat_data.extend(mpc_pred_diff_t.values)

    mpc_pred_diff_t[mpc_pred_diff_t<=loss_threshhold] = 0

    mpc_pred_diff = mpc_pred_diff_t.sum()

    mpc100_diff_t = (df_optimal['int'] - df_optimal['mpc100']).abs()
    mpc100_diff_t[mpc100_diff_t<=loss_threshhold]=0

    mpc40_diff_t = (df_optimal['int'] - df_optimal['mpc40']).abs()
    mpc40_diff_t[mpc40_diff_t <= loss_threshhold] = 0

    naive100_diff_t = (df_optimal['int'] - df_optimal['naive100']).abs()
    naive100_diff_t[naive100_diff_t <= loss_threshhold] = 0

    naive40_diff_t = (df_optimal['int'] - df_optimal['naive40']).abs()
    naive40_diff_t[naive40_diff_t <= loss_threshhold] = 0

    mpc100_diff =  mpc100_diff_t.sum()
    mpc40_diff = mpc40_diff_t.sum()
    naive100_diff = naive100_diff_t.sum()
    naive40_diff = naive40_diff_t.sum()



    index = df_pred.index[0].normalize()

    data_list.append([mpc_pred_diff,mpc100_diff, mpc40_diff, naive100_diff, naive40_diff])

    index_list.append(index)



diff_df = pd.DataFrame(data=data_list,index=index_list,columns=col_list).sort_index()

sum_df = pd.DataFrame([diff_df.sum()], index=['sum'])

print(diff_df,sum_df)

print("MEDIAN",np.median(stat_data))
print("AVG",np.mean(stat_data))
print(np.percentile(stat_data,80))
#print("MEDIAN",np.median(stat_data))
#print("MEDIAN",np.median(stat_data))

output_path = os.path.join(full_path,"energy_throughput.csv")
pd.DataFrame.to_csv(diff_df,output_path)

output_path_sum = os.path.join(full_path,"energy_throughput_sum.csv")
pd.DataFrame.to_csv(sum_df,output_path_sum)






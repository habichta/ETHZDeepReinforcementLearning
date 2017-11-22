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
args = parser.parse_args()


pred_path = args.pred_path


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
day_list = sorted([str(day) for day in day_list])


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

    rl_int_pd = pd.concat([df_optimal['int'],df_optimal['mpc100'],df_optimal['naive100'], df_pred['ci']],axis=1)


    rl_int_pd.plot()

    plt.show()









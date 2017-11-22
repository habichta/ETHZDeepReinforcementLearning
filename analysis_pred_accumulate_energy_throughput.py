import argparse
import os
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("pred_path", help="date to show")
parser.add_argument("change_constraint", help="date to show") #100
parser.add_argument("skip_pred", help="date to show") #1

args = parser.parse_args()
p_path = str(args.pred_path)
c_const = str(args.change_constraint)
skip_pred = str(args.skip_pred)

f_name = os.path.join("MPC"+str(c_const)+"-"+str(skip_pred),"energy_throughput.csv")


paths = [os.path.join(p_path,dI,f_name) for dI in os.listdir(p_path) if os.path.isdir(os.path.join(p_path,dI))]


df_list = list()

for path in paths:

    df = pd.DataFrame.from_csv(path,index_col=0,parse_dates=True)['mpc_pred100']

    parts = path.split('/')


    df = pd.DataFrame({parts[-3]: df.values})


    df_list.append(df)



full_df = pd.concat(df_list,axis=1)
sum_df = full_df.sum(axis=0)

full_df.to_csv(os.path.join(p_path,"energy_throughput_comparison.csv"))
sum_df.to_csv(os.path.join(p_path,"energy_throughput_sum.csv"))







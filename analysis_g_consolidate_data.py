#consolidate data in data_C_int folder that has been created over time (into a single dataframe


import abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline as abb_rp
import pandas as pd


day_list = abb_rp.read_cld_img_day_range_paths()

for day in day_list:
    int_path = day[1]
    cs_path = day[1].rsplit('-',1)[0]+"-cs.csv"
    mpc100_path = day[1].rsplit('-', 1)[0] + "-mpc.csv" #TODO: change to mpc100 when finished
    mpc100_new_path = day[1].rsplit('-', 1)[0] + "-mpc100.csv"
    mpc40_path = day[1].rsplit('-', 1)[0] + "-mpc40.csv"
    naive100_path = day[1].rsplit('-', 1)[0] + "-naive100.csv"
    naive100_path_new = day[1].rsplit('-', 1)[0] + "-naive100_new.csv"
    naive40_path = day[1].rsplit('-', 1)[0] + "-naive40.csv"
    all_path = day[1].rsplit('-', 1)[0] + "-all.csv"

    int_s = pd.Series.from_csv(int_path).to_frame(name="int")
    cs_s = pd.Series.from_csv(cs_path).to_frame(name="cs")
    mpc100_s = pd.Series.from_csv(mpc100_path).to_frame(name="mpc100")
    mpc100_s_new = pd.Series.from_csv(mpc100_new_path).to_frame(name="mpc100_new")
    mpc40_s = pd.Series.from_csv(mpc40_path).to_frame(name="mpc40")
    naive100_s = pd.Series.from_csv(naive100_path).to_frame(name="naive100")
    naive100_s_new = pd.Series.from_csv(naive100_path_new).to_frame(name="naive100_new")
    naive40_s = pd.Series.from_csv(naive40_path).to_frame(name="naive40")


    df_all = pd.concat([int_s,cs_s,mpc100_s,mpc100_s_new,mpc40_s,naive100_s,naive100_s_new,naive40_s],axis=1).dropna()

    pd.DataFrame.to_csv(df_all,all_path)
    
    print(df_all)


import pandas as pd
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp

from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import datetime as dt

import os

df_master = pd.read_hdf('/media/data/Daten/data_C_int/master_index_cav.h5')



file_filter={"_sp_256", ".jpeg"}

df_data_files=[]
df_label_files=[]
#(dt.datetime.strptime('2015-09-28', '%Y-%m-%d'),dt.datetime.strptime('2015-09-30', '%Y-%m-%d'))
for day in abb_rp.read_cld_img_time_range_paths(img_d_tup_l=None,automatic_daytime=True,
                                                file_filter=file_filter, get_sp_data=True,get_cs_data=True,get_mpc_data=True, randomize_days=False):

    #Img

    image_keys = list(day[0].keys())

    img_data = list(day[0].values())

    folders = [p.split('/')[5] for p in img_data]
    names = [p.split('/')[6] for p in img_data]

    img_df = pd.DataFrame(data={'folder':folders,'name':names},index=image_keys)



    #IRR
    irr = pd.read_csv(day[1] , index_col=0, parse_dates=True,
                      header=None)
    irr_data = irr.loc[pd.to_datetime(image_keys)]
    irr_data.columns = [['irradiation_hs']]


    #MPC100
    mpc= pd.read_csv(day[2].rsplit('.',1)[0]+"100.csv", index_col=0, parse_dates=True,
                               header=None)
    mpc_data = mpc.loc[pd.to_datetime(image_keys)]
    mpc_data.columns = [['mpc100']]

    mpc40 = pd.read_csv(day[2].rsplit('.', 1)[0] + "40.csv", index_col=0, parse_dates=True,
                      header=None)
    mpc_data_40 = mpc40.loc[pd.to_datetime(image_keys)]
    mpc_data_40.columns = [['mpc40']]

    #Naive100

    naive = pd.read_csv(day[2].rsplit('-', 1)[0] + "-naive100.csv", index_col=0, parse_dates=True,
                      header=None)
    naive_data = naive.loc[pd.to_datetime(image_keys)]
    naive_data.columns = [['naive100']]

    naive_new = pd.read_csv(day[2].rsplit('-', 1)[0] + "-naive100_new.csv", index_col=0, parse_dates=True,
                        header=None)
    naive_data_new = naive_new.loc[pd.to_datetime(image_keys)]
    naive_data_new.columns = [['naive100_new']]

    naive_new_40 = pd.read_csv(day[2].rsplit('-', 1)[0] + "-naive40_new.csv", index_col=0, parse_dates=True,
                            header=None)
    naive_data_new_40 = naive_new_40.loc[pd.to_datetime(image_keys)]
    naive_data_new_40.columns = [['naive40_new']]





    df_master = df_master[~df_master.index.duplicated(keep='last')]
    df_temp = df_master.loc[pd.to_datetime(image_keys)]['T1']



    print(len(df_temp),len(image_keys))



    df_t1 = pd.DataFrame(data={'T1':df_temp.values},index=image_keys)



    # Clearsky
    cs= pd.read_csv(day[3], index_col=0, parse_dates=True,
                               header=None)  # read sp file data with sunspot coordinates
    cs_data = cs.loc[pd.to_datetime(image_keys)]
    cs_data.columns = [['ghi']]




    #Sunspot coords
    sunspot_data = pd.read_csv(day[4], index_col=0, parse_dates=True,
                               header=None)  # read sp file data with sunspot coordinates
    sunspot_coords = sunspot_data.loc[pd.to_datetime(image_keys)].ix[:, 0:2]
    sunspot_coords.columns=[['sun_x','sun_y']]


    day_data_df = pd.concat([img_df,irr_data,mpc_data,mpc_data_40,naive_data,naive_data_new,naive_data_new_40,df_t1,cs_data,sunspot_coords],axis=1)

    print(day_data_df.head(1))
    df_data_files.append(day_data_df)


    #LABELS
    l_path = os.path.join(abb_c.c_img_path,folders[0],folders[0]+"-labels.csv")

    labels_df = pd.read_csv(l_path, index_col=0, parse_dates=True)


    labels_df=labels_df.loc[pd.to_datetime(image_keys)]


    irr_l_list = ["IRR"+str(i) for i in range(31)]
    #mpc_l_list = ["MPC" + str(i) for i in range(31)]
    r_l =["B","C"]

    labels = irr_l_list+r_l#+mpc_l_list

    l_df = labels_df[labels]

    print(l_df.head(1))

    df_label_files.append(l_df)





df_data_master = pd.concat(df_data_files,axis=0).sort_index()
df_data_master.index = pd.to_datetime(df_data_master.index)

df_label_master = pd.concat(df_label_files,axis=0).sort_index()
df_label_master.index = pd.to_datetime(df_label_master.index)

print(len(df_data_master.index),len(df_label_master.index))
print(len(df_data_master.index.intersection(df_label_master.index)))

print(set(df_data_master.index.date),len(set(df_data_master.index.date)))
print(set(df_label_master.index.date),len(set(df_label_master.index.date)))

df_data_master.to_hdf(os.path.join(abb_c.c_int_data_path,"C-data-master.h5"),key="data_master",mode='w',format='table',data_columns=True, compression='gzip')

#df_label_master.to_hdf(os.path.join(abb_c.c_int_data_path,"C-irr_label-master.h5"),key="label_master",mode='w',format='table',data_columns=True, compression='gzip')



"""
df_test = pd.read_hdf('/media/data/Daten/data_C_int/C-data-master.h5')

print(df_test)
"""















from abb_deeplearning.abb_mpc_controller import abb_mpc
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline import read_full_int_irr_data
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import shutil



parser = argparse.ArgumentParser()
parser.add_argument("pred_path", help="date to show")
parser.add_argument("change_constraint", help="date to show") #100
parser.add_argument("skip_pred", help="date to show") #1
parser.add_argument("date", default="",nargs="?")
parser.add_argument("--save", default="True",nargs="?")
parser.add_argument("--plot", default="False",nargs="?")
args = parser.parse_args()

pred_path = args.pred_path
change_constraint=int(args.change_constraint)
skip_preds=int(args.skip_pred)
date=str(args.date)
save=str(args.save)
plot=str(args.plot)

"""
pred_path = '/home/dladmin/Documents/arthurma/runs/RESNET7_irr_nopool_absmax_EVAL/simple_2_balFalse_typeB_regression_resnetV218_irr_ollinear11_irr_kpnodropout_1_diffFalse_ch3_isFalse_imFalse_pool_layerFalse_lfabs_max_maxw0.5_outl200_290199'
date="2015-11-08"
change_constraint=100
skip_preds=1
"""

full_path = os.path.join(pred_path,"MPC"+str(int(change_constraint))+"-"+str(int(skip_preds)))


file_paths = os.listdir(full_path)

print(file_paths)

if date is "":
    file_paths = sorted([os.path.join(full_path,path) for path in file_paths])
else:
    file_paths = sorted([os.path.join(full_path, path) for path in file_paths if date in path])

naive_value_paths = [os.path.join(ac.c_int_data_path,(os.path.basename(path.rsplit('-',1)[0]))+"-naive"+str(int(change_constraint))+".csv") for path in file_paths]
mpc_value_paths = [os.path.join(ac.c_int_data_path,(os.path.basename(path.rsplit('-',1)[0]))+"-mpc"+str(int(change_constraint))+".csv") for path in file_paths]


for day,day_naive,day_mpc in sorted(zip(file_paths,naive_value_paths,mpc_value_paths)):

    try:
        print(day,day_naive,day_mpc)

        if "mpc" in day:

            df_data = pd.DataFrame.from_csv(day)

            date = str(df_data.index[0].date())
            print(date)


            df_data_naive = pd.DataFrame.from_csv(day_naive)
            df_data_mpc = pd.DataFrame.from_csv(day_mpc)

            df_data_naive = df_data_naive.loc[df_data.index]
            df_data_mpc = df_data_mpc.loc[df_data.index]
            df_data_naive.columns = ["naive"+str(int(change_constraint))]
            df_data_mpc.columns = ["mpc" + str(int(change_constraint))]
            df_data_plot = pd.concat([df_data,df_data_naive, df_data_mpc],axis=1)

            df_data_plot.plot()
            fig = plt.gcf()

            if plot == "True":
                plt.show()



            if save == "True":


                save_folder = os.path.join(full_path,"pred_img")

                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)

                save_path = os.path.join(save_folder,date)

                fig.savefig(save_path, format="png")


    except FileNotFoundError as fe:
        print(fe)





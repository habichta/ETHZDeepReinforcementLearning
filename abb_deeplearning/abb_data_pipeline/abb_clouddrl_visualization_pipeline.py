
from . import abb_clouddrl_constants as acc
from .abb_clouddrl_constants import ABB_Solarstation
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import re
import time
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication
import matplotlib.image as mpimg
import datetime as dt
import matplotlib.dates as md

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.resize(250, 150)
        self.center()

        self.setWindowTitle('Center')
        self.show()

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


def plot_irradiance_mpc(solar_station=ABB_Solarstation.C, date_path=None):
    """
    plots the MPC and Irradiance data. Only tested for same frequencies ( interpolated MPC with 1s frequency)
    date: path to file
    """
    if date_path is None:
        raise ValueError('No date given')

    if solar_station is ABB_Solarstation.C:
        path = acc.c_int_data_path
    elif solar_station is ABB_Solarstation.MS:
        path = acc.ms_int_data_path

    mpc_path = os.path.join(os.path.dirname(date_path), os.path.basename(
        date_path).rsplit('.', 1)[0].rsplit('-', 1)[0] + '-mpc.csv')

    cs_path = os.path.join(os.path.dirname(date_path), os.path.basename(
        date_path).rsplit('.', 1)[0].rsplit('-', 1)[0] + '-cs.csv')


    int_s  = pd.Series.from_csv(
        path=date_path, sep=',', index_col=0, infer_datetime_format=True)
    mpc_s  = pd.Series.from_csv(
        path=mpc_path, sep=',', index_col=0, infer_datetime_format=True)

    cs_s  = pd.Series.from_csv(
        path=cs_path, sep=',', index_col=0, infer_datetime_format=True)

    cs_s70 = cs_s*0.7

    data = pd.DataFrame(dict(int_s=int_s, mpc_s=mpc_s,cs_s = cs_s,cs_s70=cs_s70)).reset_index()

    data = data.ix[2:]

    # drop last few lines where MPC may be NAN due to frequency issues
    data = data[np.isfinite(data['mpc_s'])]
    data.plot(x='index')
    plt.show()


def index_to_img_path(index,solar_station=ABB_Solarstation.C):
    index = str(index)
    date = index.split(" ")[0]
    index = index.strip()

    prefix = "C-" if solar_station==ABB_Solarstation.C else "MS-"

    path = os.path.join(prefix+date,re.sub(r'\D', '_', index)+"_Resize256.jpeg")


    return path


def plot_predictions_of_day(solar_station=ABB_Solarstation.C, pred_nr = 10,pred_path=None,date=None):
    if pred_path is None:
        raise ValueError('No path')

    if solar_station is ABB_Solarstation.C:
        path_int = acc.c_int_data_path
        path_img = acc.c_img_path
    elif solar_station is ABB_Solarstation.MS:
        path_int= acc.ms_int_data_path
        path_img = acc.ms_img_path

    predictions = pd.DataFrame.from_csv(os.path.join(pred_path,"eval_predictions.csv"))

    predictions = predictions[(predictions.index.get_level_values(0) >= date)]

    for index,row in predictions.iterrows():
        #print(index_to_img_path(index,solar_station),row)

        img_path = os.path.join(path_img,index_to_img_path(index,solar_station))

        img = mpimg.imread(img_path)

        pred = row["P0":"P"+str(int(pred_nr))]
        label = row["L0":"L"+str(int(pred_nr))]

        x = np.linspace(0, len(pred.index), len(pred.index))
        f, axarr = plt.subplots(2)
        axarr[0].imshow(img)
        axarr[0].set_title(str(index))
        axarr[1].plot(x,label,label="label")
        axarr[1].plot(x, pred, label="pred")
        axarr[1].set_ylim(0,1200)
        handles,plot_labels = axarr[1].get_legend_handles_labels()
        axarr[1].legend(handles, plot_labels)


        plt.show()



def save_plot_predictions_of_day(solar_station=ABB_Solarstation.C, pred_nr = 10, pred_path=None,dates=None):

   """
   :param solar_station: ...
   :param pred_path: path to eval folder
   :param dates: list of yyyy-mm-dd hh-mm-ss
   :return: 
   """
   if pred_path is None:
       raise ValueError('No path')

   if solar_station is ABB_Solarstation.C:
       path_int = acc.c_int_data_path
       path_img = acc.c_img_path
   elif solar_station is ABB_Solarstation.MS:
       path_int = acc.ms_int_data_path
       path_img = acc.ms_img_path

   predictions = pd.DataFrame.from_csv(os.path.join(pred_path, "eval_predictions.csv"))
   path = os.path.join(pred_path, "sample_img")

   if not os.path.exists(path):
       os.makedirs(path)

   for date in dates:
       try:
           name = date + ".png"
           save_path = os.path.join(path, name)
           if not os.path.isfile(save_path):
               pred_df = predictions[(predictions.index.get_level_values(0) == date)]
               print(pred_df)
               index,row = next(pred_df.iterrows())
               img_path = os.path.join(path_img, index_to_img_path(index, solar_station))
               img = mpimg.imread(img_path)
               pred = row["P0":"P"+str(int(pred_nr))]
               label = row["L0":"L"+str(int(pred_nr))]
               x = np.linspace(0, len(pred.index), len(pred.index))
               f, axarr = plt.subplots(2)
               axarr[0].imshow(img)
               axarr[0].set_title(str(index))
               axarr[1].plot(x, label, label="label")
               axarr[1].plot(x, pred, label="pred")
               axarr[1].set_ylim(0, 1200)
               handles, plot_labels = axarr[1].get_legend_handles_labels()
               axarr[1].legend(handles, plot_labels)



               plt.savefig(save_path, format="png")
               plt.close()
       except Exception as ve:
           print("Day does not exists in predictions")



def analyze_bad_samples(pred_path=None):

   """
   analyze samples in eval_bad_predictions.csv
   :param solar_station: ...
   :param pred_path: path to eval folder
   :param dates: list of yyyy-mm-dd hh-mm-ss
   :return: 
   """
   if pred_path is None:
       raise ValueError('No path')

   count_list = list()
   count_h_list = list()

   for path in pred_path:
       try:
           df_f= pd.read_csv(os.path.join(path, "eval_bad_predictions.csv"),index_col=0,parse_dates=True )

           df= df_f["P0"]
           counts = df.groupby([df.index.year, df.index.month, df.index.day]).count()
           counts_h = df.groupby([df.index.year, df.index.month, df.index.day,df.index.hour]).count()

           idx = pd.to_datetime(counts.index.map(lambda x: '-'.join((str(x[0]), str(x[1]),str(x[2])))),format="%Y-%m-%d")
           idx_h = pd.to_datetime(counts_h.index.map(lambda x: '-'.join((str(x[0]), str(x[1]), str(x[2]),str(x[3])))), format="%Y-%m-%d-%H")


           counts = pd.DataFrame({os.path.basename(path): counts.values})
           counts_h = pd.DataFrame({os.path.basename(path): counts_h.values})
           counts.index = idx
           counts_h.index = idx_h

           counts.to_csv(os.path.join(path,"eval_bad_predictions_analysis_day.csv"))
           counts_h.to_csv(os.path.join(path, "eval_bad_predictions_analysis_hour.csv"))
           count_list.append(counts)
           count_h_list.append(counts_h)
       except FileNotFoundError as e:
           pass

   concat_count = pd.concat(count_list,axis=1)
   concat_count_h = pd.concat(count_h_list, axis=1)

   par_dir = os.path.abspath(os.path.join(path, os.pardir))

   print("PARDIR",par_dir)
   concat_count.to_csv(os.path.join(par_dir, "eval_bad_predictions_analysis_day_comparison.csv"))
   concat_count_h.to_csv(os.path.join(par_dir, "eval_bad_predictions_analysis_hour_comparison.csv"))

   print(concat_count)
   print(concat_count_h)


def plot_loss_predictions_of_day(solar_station=ABB_Solarstation.C, path=None,date=None):


    if date:

        date_from = date.replace(hour=7,minute=0,second=0)
        date_to = date.replace(hour=23,minute=59,second=59)
        print(date_from,date_to)
        df = pd.DataFrame.from_csv(os.path.join(path,"eval_prediction_losses.csv"))
        #print(df.index)
        df.index = pd.to_datetime(df.index)

        df = df.loc[(df.index <= date_to) & (df.index>= date_from)].reset_index()

        df.plot(x='index')
    else:
        df = pd.DataFrame.from_csv(os.path.join(path, "eval_prediction_losses.csv")).plot()

    #ax = df.plot(xticks=df.index)
    #ax.xaxis.set_major_formatter(md.DateFormatter('%d:%d'))

    #plt.plot(df.index,df['MSE Loss'])
    plt.show()


    #TODO finish, think about how to plot predictions
    

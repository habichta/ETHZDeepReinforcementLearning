from ..abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
from ..abb_data_pipeline import abb_clouddrl_constants as abb_c

import cvxpy as cvx
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import random
import math
import matplotlib.pyplot as plt
import scipy
import os
import datetime as dt


def perform_default_mpc(day_path_list, time_range=None, resolution_s=1, change_constraint_wh_min=abb_c.solar_irradiance / 10, output_path=None, interpolate_to_s=False, plot=False, write_file=False):
    """
    Multithreaded calculation of Model Predictive control (calls__default_mpc__ function) with default parameters. Order of day_path_list (containing paths to files that are fromatted like this: (datetime,irradiance data))
    may be lost, depending on thread scheduling. Set output path. If None, same output path as path to the input files
    time_range: tuple of datetime formatted to and from values (to,from), use strptime without the time() method, if None, the whole time span will be used
    resolution_s: sampling rate in seconds. Lower: More accurate but slower, may not work if too low.too
    change_constraint_wh_min: change of solar irradiance measured in Watts per square meter. Maximum change per Minute (constraint)
    output_path: Where data is printed
    interpolate_to_s: Interpolate the data to second frequency (linear interpolation)
    plot: Create plot of LP data. Only works when  __default_mpc__ is called directly

    writes file with Pandas TimeSeries data

    GLPK Solver needs to be installed for cvxpy!
    """

    with ThreadPoolExecutor(max_workers=1) as thread_executor:
        [thread_executor.submit(__default_mpc__, day_path, time_range, resolution_s, change_constraint_wh_min, output_path, interpolate_to_s, plot, write_file)
         for day_path in day_path_list]




def __default_mpc__(day_path, time_range=None, resolution_s=1, change_constraint_wh_min=abb_c.solar_irradiance / 10,
                    output_path=None, interpolate_to_s=False, plot=False, write_file=False):
    # Load data into pandas Series
    print("solving:",day_path)
    if time_range is None:
        target_data = pd.Series.from_csv(
            path=day_path, sep=',', index_col=0, infer_datetime_format=True)
        time_range = (target_data.index[0], target_data.index[-1])
    else:
        target_data = pd.Series.from_csv(
            path=day_path, sep=',', index_col=0, infer_datetime_format=True).between_time(time_range[0].time(),
                                                                                          time_range[1].time())

    # Define MPC variables
    target_states_fullres = target_data.values
    resolution_steps = np.arange(0, target_states_fullres.size, resolution_s)

    target_states = target_states_fullres[resolution_steps]
    time_horizon = target_states.size

    max_change, min_change = cvx.Parameter(
        time_horizon - 1), cvx.Parameter(time_horizon - 1, sign='negative')
    max_change.value, min_change.value = np.ones(
        time_horizon - 1) * (change_constraint_wh_min * (resolution_s / 60)), np.ones(time_horizon - 1) * -(
    change_constraint_wh_min * (resolution_s / 60))
    target_val_param = cvx.Parameter(time_horizon)
    target_val_param.value = target_states

    print('Setting up constraints...')
    # Constraints and Costs. A is a Band matrix with [-1,1] encoding the
    # temporal difference between state values
    x = cvx.Variable(time_horizon)
    cost = cvx.sum_entries(cvx.abs(x - target_val_param))

    A = np.zeros((time_horizon - 1, time_horizon))
    np.fill_diagonal(A, -1)
    np.fill_diagonal(A[:, 1:], 1)
    A = scipy.sparse.csr_matrix(A)
    constr = [A * x <= max_change, A * x >= min_change, x >= 0]

    print('Solving ...')
    cvx.Problem(cvx.Minimize(cost), constr).solve(solver=cvx.GLPK)

    # Prepare Pandas output
    date = dt.datetime.strptime(
        '-'.join(os.path.basename(day_path).split('-')[1:4]), '%Y-%m-%d')
    d_from = dt.datetime.combine(date.date(), time_range[0].time())
    d_to = dt.datetime.combine(date.date(), time_range[1].time())

    dti = pd.DatetimeIndex(start=d_from, end=d_to,
                           freq=str(resolution_s) + 'S')

    sol_pd = pd.Series.from_array(np.squeeze(
        np.asarray(x.value)), index=dti)  # SHAPE PROBLEM

    if plot:
        plt.plot(resolution_steps, sol_pd)
        plt.plot(np.arange(0, target_states_fullres.size),
                 target_states_fullres)
        plt.show()

    if interpolate_to_s:
        sol_pd = sol_pd.asfreq('S')
        sol_pd = sol_pd.astype(float).interpolate(method='time')

    if write_file:
        new_filename = '-'.join(
            (os.path.basename(day_path.rsplit('-', 1)[0]), 'mpc'+str(int(change_constraint_wh_min))))
        if output_path is None:
            file_path = os.path.dirname(day_path)
            output_path_t = os.path.join(file_path, new_filename)
        else:
            output_path_t = os.path.join(output_path, new_filename)

        __mpc_file_printer__(sol_pd, output_path_t)


def __prediction_mpc__(prediction_path,day_list,solar_station=abb_c.ABB_Solarstation.C,nr_predictions = 11,pred_interval_s=60, change_constraint_wh_min=abb_c.solar_irradiance / 10,full_int_irr_pd=None,skip_preds=1,
                    output_path=None, plot=False, write_file=False):


    if solar_station==abb_c.ABB_Solarstation.C:
        suffix = "C-"
    elif solar_station==abb_c.ABB_Solarstation.MS:
        suffix = "MS-"
    else:
        raise ValueError("Illegal solar station")



    resolution_s = 1
    # Load data into pandas Series
    full_path = os.path.join(prediction_path,"eval_predictions.csv")
    prediction_data_full = pd.DataFrame.from_csv(path=full_path,sep=',', index_col=0, infer_datetime_format=True)
    prediction_data_full = prediction_data_full.sort_index()

    #prediction_data_full = prediction_data_full.ix['2015-07-24 05:00:00':'2016-07-24 23:00:00']
    #day_list=['2015-07-24']


    for day in day_list:

        try:
            prediction_data = prediction_data_full.loc[day]
        except KeyError as ke:
            print(ke)
            continue



        mpc_data_list = list()
        label_data_list=list()
        pred_data_list = list()

        first_row = prediction_data.iloc[0]['P0']
        last_control_input = first_row # this is the first "control input level", initialize to the very first prediction of current irradiance

        #Initialize iterators
        row_iterator = prediction_data.iterrows()
        row_iterator_n = prediction_data.iterrows()
        first_iteration = True
        for i in range(skip_preds):
            next(row_iterator_n)

        last_prediction = 'P'+str(nr_predictions-1)
        last_label = 'L' + str(nr_predictions - 1)
        counter = 1

        for ((ind, row),(ind_n,row_n)) in zip(row_iterator,row_iterator_n): # go through the whole prediction file, row by row, each row looks approx 6 to 8 seconds further into the future

            print("SEC DIFF",(ind_n - ind).total_seconds())


            #Some samples have missing values. the predictions jump over them but when calculating predictions they are not ignored and usullay these gaps produce a large error afterwards
            if (ind_n-ind).total_seconds() > 2*8*skip_preds:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR IN DATA, SKIP ", ind)
                continue
            if not first_iteration:
                if counter < skip_preds:
                    counter+=1
                    continue
            counter = 1
            first_iteration = False

            pred = row['P0':last_prediction]
            label = row['L0':last_label]


            new_index = pd.date_range(ind,periods=nr_predictions,freq=str(pred_interval_s)+"S")
            new_index_next = pd.date_range(ind_n,periods=nr_predictions,freq=str(pred_interval_s)+"S")

            print("INDICES", ind,ind_n)


            pred_pd = pd.DataFrame(data=pred.values,index=new_index)
            pred_pd = pred_pd.asfreq(str(resolution_s)+'S')
            target_data = pred_pd.astype(float).interpolate(method='time')


            label_pd = pd.DataFrame(data=label.values, index=new_index)
            label_pd = label_pd.asfreq(str(resolution_s)+'S')
            label_data = label_pd.astype(float).interpolate(method='time')

            # Define MPC variables
            target_states = target_data.values #linearly interpolated according to P0 to P20 predictions
            #resolution_steps = np.arange(0, target_states_fullres.size, resolution_s)

            #target_states = target_states_fullres[resolution_steps]
            time_horizon = target_states.size

            max_change, min_change = cvx.Parameter(
                time_horizon - 1), cvx.Parameter(time_horizon - 1, sign='negative')
            max_change.value, min_change.value = np.ones(
                time_horizon - 1) * (change_constraint_wh_min * (resolution_s / 60)), np.ones(time_horizon - 1) * -(
                change_constraint_wh_min * (resolution_s / 60))  #default change_constraint_wh_min is 100 per minute)
            target_val_param = cvx.Parameter(time_horizon)
            target_val_param.value = target_states

            last_control_max,last_control_min = cvx.Parameter(1),cvx.Parameter(1)
            last_control_max.value = last_control_input+(change_constraint_wh_min * (resolution_s / 60)) # level of the MPC value of the last step
            last_control_min.value = last_control_input-(change_constraint_wh_min * (resolution_s / 60))
            print('Setting up constraints...')
            # Constraints and Costs. A is a Band matrix with [-1,1] encoding the
            # temporal difference between state values
            x = cvx.Variable(time_horizon)
            cost = cvx.sum_entries(cvx.abs(x - target_val_param))

            A = np.zeros((time_horizon - 1, time_horizon))
            np.fill_diagonal(A, -1)
            np.fill_diagonal(A[:, 1:], 1)
            A = scipy.sparse.csr_matrix(A)
            constr = [A * x <= max_change, A * x >= min_change, x >= 0,x[0]<=last_control_max,x[0]>=last_control_min]

            print('Solving ...')
            cvx.Problem(cvx.Minimize(cost), constr).solve(solver=cvx.GLPK)


            # Prepare Pandas output

            mpc_pd = pd.Series.from_array(np.squeeze(
                np.asarray(x.value)), index=target_data.index)  # SHAPE PROBLEM

            #print(target_data,mpc_pd)


            mpc_step = mpc_pd.loc[mpc_pd.index < new_index_next[0]] #MPC step will be the step for this iterartion until the next (usually around 6-8 seconds difference
             #this is an effective step
            last_control_input = mpc_step[-1] # the last value of an effective mpc step. This will constrain the first mpc value of the next step into the legal ramp
            pred_step = target_data.loc[mpc_step.index.values]
            label_step = label_data.loc[mpc_step.index.values]

            print(pred_step,mpc_step)

            mpc_data_list.append(mpc_step)
            pred_data_list.append(pred_step)
            label_data_list.append(label_step)


            if plot:
                plt.plot(np.arange(0, len(mpc_pd.index.values)), mpc_pd, label="pred_mpc")
                plt.plot(np.arange(0, len(target_data.index.values)),
                         target_data.values,label='pred')
                plt.plot(np.arange(0, len(label_data.index.values)),
                         label_data,label="label")
                #plt.ylim([200,350])
                plt.legend()
                plt.show()
                plt.close()


        mpc_data_all = pd.concat(mpc_data_list)
        label_data_all = pd.concat(label_data_list)
        pred_data_all = pd.concat(pred_data_list)


        print(label_data_all)


        if full_int_irr_pd is not None:
            irr_data_all = full_int_irr_pd.loc[mpc_data_all.index.values]
            full_data = pd.concat([mpc_data_all,pred_data_all,label_data_all,irr_data_all],axis=1)
            full_data.columns=["mpc_pred","pred","labels","actual_irr"]
        else:
            full_data = pd.concat([mpc_data_all, pred_data_all, label_data_all], axis=1)
            full_data.columns = ["mpc_pred", "pred", "labels"]


        output_path = os.path.join(prediction_path,"MPC"+str(int(change_constraint_wh_min)))+"-"+str(skip_preds)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = suffix+day+"-pred_mpc"+str(int(change_constraint_wh_min))


        output_path = os.path.join(output_path,filename)
        __mpc_file_printer__(full_data, output_path)

        #full_data.plot()
        #plt.show()


def __prediction_mpc_optimal__(prediction_path,pred_interval_s=60, change_constraint_wh_min=abb_c.solar_irradiance / 10,full_int_irr_pd=None,
                    output_path=None, plot=False, write_file=False):
    resolution_s = 1
    # Load data into pandas Series
    full_path = os.path.join(prediction_path, "eval_predictions.csv")
    prediction_data = pd.DataFrame.from_csv(path=full_path, sep=',', index_col=0, infer_datetime_format=True)

    #prediction_data = prediction_data.ix['2015-07-16 07:43:20':'2015-07-16 8:45:20']

    mpc_data_list = list()
    label_data_list = list()
    pred_data_list = list()

    first_row = prediction_data.iloc[0]['P0']
    last_control_input = first_row  # this is the first "control input level", initialize to the very first prediction of current irradiance

    row_iterator = prediction_data.iterrows()
    row_iterator_n = prediction_data.iterrows()
    next(row_iterator_n)

    for ((ind, row), (ind_n, row_n)) in zip(row_iterator,
                                            row_iterator_n):  # go through the whole prediction file, row by row, each row looks approx 6 to 8 seconds further into the future
        pred = row['P0':'P10']

        new_index = pd.date_range(ind, periods=11, freq=str(pred_interval_s) + "S")
        new_index_next = pd.date_range(ind_n, periods=11, freq=str(pred_interval_s) + "S")

        pred_pd = pd.DataFrame(data=pred.values, index=new_index)
        pred_pd = pred_pd.asfreq(str(resolution_s) + 'S')
        target_data = pred_pd.astype(float).interpolate(method='time')

        target_data = full_int_irr_pd.loc[target_data.index]
        
        # Define MPC variables
        target_states = target_data.values  # linearly interpolated according to P0 to P20 predictions
        # resolution_steps = np.arange(0, target_states_fullres.size, resolution_s)

        # target_states = target_states_fullres[resolution_steps]
        time_horizon = target_states.size

        max_change, min_change = cvx.Parameter(
            time_horizon - 1), cvx.Parameter(time_horizon - 1, sign='negative')
        max_change.value, min_change.value = np.ones(
            time_horizon - 1) * (change_constraint_wh_min * (resolution_s / 60)), np.ones(time_horizon - 1) * -(
            change_constraint_wh_min * (resolution_s / 60))  # default change_constraint_wh_min is 100 per minute)
        target_val_param = cvx.Parameter(time_horizon)
        target_val_param.value = target_states

        last_control_max, last_control_min = cvx.Parameter(1), cvx.Parameter(1)
        last_control_max.value = last_control_input + (
        change_constraint_wh_min * (resolution_s / 60))  # level of the MPC value of the last step
        last_control_min.value = last_control_input - (change_constraint_wh_min * (resolution_s / 60))
        print('Setting up constraints...')
        # Constraints and Costs. A is a Band matrix with [-1,1] encoding the
        # temporal difference between state values
        x = cvx.Variable(time_horizon)
        cost = cvx.sum_entries(cvx.abs(x - target_val_param))

        A = np.zeros((time_horizon - 1, time_horizon))
        np.fill_diagonal(A, -1)
        np.fill_diagonal(A[:, 1:], 1)
        A = scipy.sparse.csr_matrix(A)
        constr = [A * x <= max_change, A * x >= min_change, x >= 0, x[0] <= last_control_max, x[0] >= last_control_min]

        print('Solving ...')
        cvx.Problem(cvx.Minimize(cost), constr).solve(solver=cvx.GLPK)

        # Prepare Pandas output

        mpc_pd = pd.Series.from_array(np.squeeze(
            np.asarray(x.value)), index=target_data.index)  # SHAPE PROBLEM

        # print(target_data,mpc_pd)


        mpc_step = mpc_pd.loc[mpc_pd.index < new_index_next[
            0]]  # MPC step will be the step for this iterartion until the next (usually around 6-8 seconds difference
        # this is an effective step
        last_control_input = mpc_step[
            -1]  # the last value of an effective mpc step. This will constrain the first mpc value of the next step into the legal ramp
        pred_step = target_data.loc[mpc_step.index.values]

        print(pred_step, mpc_step)

        mpc_data_list.append(mpc_step)
        pred_data_list.append(pred_step)

        if plot:
            plt.plot(np.arange(0, len(mpc_pd.index.values)), mpc_pd, label="pred_mpc")
            plt.plot(np.arange(0, len(target_data.index.values)),
                     target_data.values, label='pred')
            plt.legend()
            plt.show()
            plt.close()

    mpc_data_all = pd.concat(mpc_data_list)
    pred_data_all = pd.concat(pred_data_list)

    if full_int_irr_pd is not None:
        irr_data_all = full_int_irr_pd.loc[mpc_data_all.index.values]
        full_data = pd.concat([mpc_data_all, pred_data_all, irr_data_all], axis=1)
        full_data.columns = ["mpc_pred", "pred", "actual_irr"]
    else:
        full_data = pd.concat([mpc_data_all, pred_data_all], axis=1)
        full_data.columns = ["mpc_pred", "pred"]

    full_data.plot()
    plt.show()

def __mpc_file_printer__(mpc_data, output_path):
    print('Print: ' + output_path)
    mpc_data.to_csv(output_path + '.csv', index=True)

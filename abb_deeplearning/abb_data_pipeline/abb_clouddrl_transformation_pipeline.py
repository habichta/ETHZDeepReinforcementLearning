'''
Contains functions to simplify transofrmation of the input data. This can be illuminance data or cloud images
Author: Arthur Habicht habichta@ethz.ch
'''

import os
import re
from .abb_clouddrl_constants import ABB_Solarstation as abb_st
from .abb_clouddrl_constants import abb_filepattern
from . import abb_clouddrl_constants as ac
from . import abb_clouddrl_read_pipeline as abb_rp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import math
import skimage.io as skio
import scipy.misc as misc
import skimage.transform as skt
import pathlib
import warnings
from collections import Counter


def abb_linear_interpolate_illuminance(path, output_path=None, filename_pattern=abb_filepattern, abb_solarstation=None,
                                       error_log=False):
    '''
    Interpolates the illuminance data and outputs interpolated data in file "input_filename"_interpolated.log
    Data is interpolated linearly with a per second resolution
    Arg: path to the log files
    Filename Pattern: default pattern is "{Any character}-year-month-day.log"
    abb_solarstation: used to parse the files of either MS or C solarstation. Adapt this if the format of the illuminance data has changed
    Returns: Void, write to output file in same folder. Prints to a csv file
    '''

    if filename_pattern is None:
        print("Using default pattern for file identification")

    if output_path is None:
        output_path = path

    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(
        os.path.join(path, f)) and filename_pattern.search(f) is not None]

    print("Nr of files found:" + str(len(files)))
    # extract illuminance data for each day, interpolate the data linearly
    for file in files:

        print("Parsing " + file)
        interpolated_data = None
        with open(file) as f:

            if abb_solarstation is abb_st.C:
                interpolated_data = __linear_interpolate_c__(f, error_log)
            elif abb_solarstation is abb_st.MS:
                interpolated_data = __linear_interpolate_ms__(f, error_log)
            else:
                raise ValueError(
                    "ABB solar power station not valid. Input Enum C or MS")

        new_filename = '-'.join((os.path.basename(file.split('.')[0]), 'int'))
        file_path = os.path.dirname(file)

        if output_path is None:
            output_path_t = os.path.join(file_path, new_filename)
        else:
            output_path_t = os.path.join(output_path, new_filename)

        __interpolated_illuminance_file_printer__(
            interpolated_data, output_path_t)


def __linear_interpolate_ms__(file, error_log):
    # Danger! Potential Bug in output format foe date time. Change similar to
    # interpolate_c explicit format with strftime!
    date_numbers = [str(t) for t in re.split(r'[.-]+', os.path.basename(
        file.name)) if t.isdigit()]
    date = '-'.join(date_numbers)

    day_illuminance_dict = {}

    for line in file:
        try:
            line_data = line.split(',')
            key = ' '.join(
                (date, ':'.join((line_data[0], line_data[1], line_data[2]))))
            value = line_data[4]

            day_illuminance_dict[key] = value

        except IndexError:
            print(
                'Experienced index error. Possibly faulty data or wrong station input! (Check!)')
            if error_log:
                with open(os.path.join(os.path.dirname(file.name), 'error_log_MS.log'), 'a') as ef:
                    ef.write("Index Error in: " +
                             os.path.basename(file.name) + '\n')

    return __linear_interpolate__(day_illuminance_dict)


def __linear_interpolate_c__(file, error_log):
    # int_d = pd.DataFrame(columns=('Time', 'Illuminance'))
    day_illuminance_dict = {}

    for line in file:
        try:
            line_data = line.split()
            key = ' '.join((line_data[0], line_data[1]))
            # internal format ist not same as file format
            key_dt = dt.datetime.strptime(key, '%d-%m-%Y %H:%M:%S')
            # reformat time so that it fits naming of file
            key_dt.strftime('%Y-%m-%d %H:%M:%S')

            value = line_data[4]
            day_illuminance_dict[key_dt] = value

        except IndexError:
            print(
                'Experienced index error. Possibly faulty data or wrong station input! (Check!)')
            if error_log:
                with open(os.path.join(os.path.dirname(file.name), 'error_log_C.log'), 'a') as ef:
                    ef.write("Index Error in: " +
                             os.path.basename(file.name) + '\n')

    return __linear_interpolate__(day_illuminance_dict)


def __linear_interpolate__(illuminance_dict):
    illuminance_ts = pd.Series(illuminance_dict)
    illuminance_ts.index = pd.to_datetime(illuminance_ts.index)
    illuminance_ts = illuminance_ts.asfreq('S')
    illuminance_ts_int = illuminance_ts.astype(
        float).interpolate(method='time')

    # illuminance_ts_int.plot()
    # plt.show()

    return (illuminance_ts_int)


def __interpolated_illuminance_file_printer__(interpolated_data_ts, output_path):
    print('Print: ' + output_path)

    interpolated_data_ts.to_csv(output_path + '.csv', index=True)


##########################################################################


def abb_create_irradiance_statistics(path, img_d_tup_l=None, output_path=None, abb_solarstation=None,
                                     print_to_csv=False):
    warnings.warn("deprecated", DeprecationWarning)
    """
    Does not use new automatic daytime mechanism that uses the clear sky model!!
    """
    if output_path is None:
        output_path = path

    if img_d_tup_l is not None:
        files = [data_tuple[1] for data_tuple in abb_rp.read_cld_img_day_range_paths(
            solar_station=abb_solarstation, img_d_tup_l=img_d_tup_l, randomize_days=False)]
        print(files)
    else:
        if abb_solarstation is abb_st.C:
            files = [data_tuple[1] for data_tuple in abb_rp.read_cld_img_day_range_paths(
                solar_station=abb_solarstation, img_d_tup_l=[(ac.c_min_date, ac.c_max_date)], randomize_days=False)]
        elif abb_solarstation is abb_st.MS:
            files = [data_tuple[1] for data_tuple in abb_rp.read_cld_img_day_range_paths(
                solar_station=abb_solarstation, img_d_tup_l=[(ac.ms_min_date, ac.ms_max_date)], randomize_days=False)]
        else:
            raise ValueError('ABB Solarstation not found (C or MS?)')

    print("Nr of files found:" + str(len(files)))
    # extract illuminance data for each day, interpolate the data linearly

    __irradiance_statistics__(files, abb_solarstation=abb_solarstation, output_path=output_path,
                              print_to_csv=print_to_csv)


def __irradiance_statistics__(interpolated_files, abb_solarstation=abb_st.C, output_path=None, print_to_csv=False):
    """
    Create dataframe that contains statistics for each day, the daytimes are cropped
    to relevant times between sunrise and sunset (approx) for each month, as defined in the
    abb constants module (for each solar power plant)
    Statistics: count,mean,std,min,percentiles(10% steps), max, std_m (squared average of rolling variance),
    histogram of nr. of seconds having a certain value with bucketsize 50 from 0 to 1000 (irradiance values)
    """
    col_stats = None
    index_dates = list()
    data = list()

    if output_path is None:
        raise ValueError('Not a valid output path for __irradiance_statistics__')

    for interpolated_file in interpolated_files:
        illuminance_ts = pd.Series.from_csv(
            interpolated_file, sep=',', index_col=0, infer_datetime_format=True)

        month = '{:02d}'.format(illuminance_ts.index[0].month)

        if abb_solarstation is abb_st.C:
            time_from, time_to = ac.c_sunrise_sunset[month]
        elif abb_solarstation is abb_st.MS:
            raise ValueError(
                'MS Not yet implemented in __irradiance_statistics__')

        # Crop the data used between sunrise and sunset of a particular month
        illuminance_ts = illuminance_ts.between_time(
            start_time=time_from.time(), end_time=time_to.time())
        index_dates.append(illuminance_ts.index[0])
        statistics = illuminance_ts.astype(float).describe(
            percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        moving_std = illuminance_ts.rolling(window=500, center=False).var()
        moving_std_mean = pd.Series(
            [math.sqrt(moving_std.mean())], index=['std_m'])
        statistics = statistics.append(moving_std_mean, verify_integrity=True)
        hist, bin_edges = np.histogram(illuminance_ts, range(0, 1050, 50))
        histogr = pd.Series(hist, index=['h' + str(bins)
                                         for bins in bin_edges[1:]])

        statistics = statistics.append(histogr)
        data.append(statistics.values)

        if col_stats is None:
            col_stats = statistics.index.tolist()

        """
        data = pd.DataFrame(dict(illuminance_ts=illuminance_ts, moving_std=np.sqrt(moving_std))).reset_index()
        # drop last few lines where MPC may be NAN due to frequency issues
        data = data[np.isfinite(data['moving_std'])]
        data.plot(x='index')

        illuminance_ts.hist()
        plt.show()

        # plt.show()
        # print(moving_std)
        #count, division = np.histogram(illuminance_ts)
        # create dataframe with all data. Check which histogram bin make sense!
     
        """

    data_df = pd.DataFrame(data=np.array(data), index=np.array(index_dates), columns=np.array(col_stats))

    if print_to_csv is True:
        data_df.to_csv(output_path, sep=',')









def abb_neural_network_labels_generator(solar_station=abb_st.C, img_d_tup_l=None, img_t_tup_l=None,
                                        automatic_daytime=False, file_filter={"Debevec", ".jpeg"}, look_ahead_sec=600,
                                        irr_mpc_value_freq_s=60,regression_data_only=True, balanced_setting=2, print_to_csv=False):
    """
    Generates labels for supervised neural network training (for each image within a time span). This includes the value function given
    The policy generated by the MPC under the assumption that the battery charge is infinite. This is the integral of
    the difference between MPC policy and actual solar irradiance. At any given picture (looking 10 minutes ahead as default) 
    Furthermore, it returns the MPC values at an image's timestamp (+10 minutes ahead, freq of 1 minute) which would indicate the optimal actions. Also it returns the irradiance
    values for the next 10 minutes (1 minute frequency as default; same as MPC values)
    It creates classification labels sunny(1) cloudy (0) using clear sky model data. if irradiance is lower than 70% of clear sky model then it is considered cloudy
    
    :argument
    img_d_tup_l: list of day ranges (datetime object tuples), all days if None
    img_t_tup_l: list of time ranges within days (datetime object tuples), between sunrise and sunset of particular powerplant given in abb_constants
    file_filter: filter strings for image files to be considered
    look_ahead_sec: How many seconds to look ahead from image time stamps (for value function and irradiance values). Default is 10 minutes. Images that do not have 10 minutes of data ahead of them are not considered
    irr_mpc_value_freq_s: The frequency of irr/MPC values within look_ahead_sec (default 600/60 = 10, one value each minute, starting from 0 seconds ahead)
    automatic_daytime: use automatic sunrise and sunset times defined in abb constants for a particular power plant (overrides img_t_tup_l)
    balanced_setting: How many changes from sunny to cloudy or vice versa are needed in order to get a 1 (changes) or 0 (no changes/too few changes)
    This is needed to calculate a balanced dataset, since most samples do not have any drastic changes from sunny to cloudy or vice versa
    In order to learn interesting samples, a balanced dataset may be useful. The number determines the minimum amount of changes. The larger the number the fewer
    samples are considered "changing". This creates an additional field in the labels csv. Tensorflow can then caluclate a weighted loss for each sample within the training set (which may change!)
    print_to_csv = True prints dataframe into image folder
    
    :return: Default: file with Image timestamp: value function label,MPC values,
    """

    # data_tuple = (dict(time->image_path), data_path_for_day) generator
    label_list = None
    for data_tuple in abb_rp.read_cld_img_time_range_paths(solar_station=solar_station, img_d_tup_l=img_d_tup_l,
                                                           img_t_tup_l=img_t_tup_l, automatic_daytime=automatic_daytime,
                                                           file_filter=file_filter, get_mpc_data=True, get_cs_data=True,
                                                           randomize_days=False):

        # print(len(list(data_tuple[0].values())),list(data_tuple[0].keys())[0],list(data_tuple[0].keys())[-1])
        # print(data_tuple[2])

        irr_data = pd.Series.from_csv(data_tuple[1])
        mpc_data = pd.Series.from_csv(data_tuple[2])
        cs_data = pd.Series.from_csv(data_tuple[3])

        diff = pd.Series.abs(irr_data - mpc_data)
        diff = diff.dropna()

        # timestep classified as sunny if irradiation is 70% or above of the clear sky model data
        sunny_labels = (irr_data >= 0.7 * cs_data) * 1



        index_list = list()
        data_list = list()

        for image_key in list(data_tuple[0].keys()):  # go through all images of the day

            # Create time range


            begin = dt.datetime.strptime(image_key, '%Y-%m-%d %H:%M:%S')
            end = begin + dt.timedelta(seconds=look_ahead_sec)

            if diff.index[-1] >= end:  # as long as the interval fits within the data
                index_list.append(begin)
                # Very inefficient, could save current sum and only add next second ...oh well..:
                value_function = diff.between_time(start_time=begin.time(), end_time=end.time()).sum()

                intervals = int(divmod((end - begin).total_seconds(), irr_mpc_value_freq_s)[
                                    0])  # discretize the interval into equal steps

                timestamps = [begin + i * dt.timedelta(seconds=irr_mpc_value_freq_s) for i in range(intervals + 1)]
                # Add to data: valuefunction, irr,mpc for each image timestamp
                """
                print("Nr labels (now + nr of predictions):",len(timestamps))
                print("From/to:", str(timestamps[0]),str(timestamps[-1]))
                print("Step difference:", str((timestamps[1]-timestamps[0]).total_seconds()))
                """

                class_labels = sunny_labels[timestamps].values # classifies 1 = sunny  0 = cloudy

                change_labels = [abs(f_i - s_i) for f_i, s_i in zip(class_labels[1:], class_labels[:-1])] # calculates changes from sunny to cloudy or cloudy to sunny both are a 1
                change_labels.insert(0,0) #assumes no change for first (current time prediction)

                change_number_label = sum(change_labels) # counts number of changes from sunny to cloudy

                c = Counter(class_labels[1:])  # count class occurences after now label
                nr_values = len(class_labels[1:])
                current_class = class_labels[0]  # sunny ord cloudy now

                balanced_label = 0

                if nr_values - c[
                    current_class] >= balanced_setting:  # if at least balanced_setting changes of cloudy to sunny or sunny to cloudy
                    balanced_label = 1
                #print(image_key)

                #print(np.array([value_function]), irr_data[timestamps].values, mpc_data[timestamps].values,
                     #sunny_labels[timestamps].values, np.array([balanced_label]))

                if not regression_data_only:
                    data_list.append(np.concatenate(
                        (np.array([value_function]), irr_data[timestamps].values, mpc_data[timestamps].values,
                         sunny_labels[timestamps].values, change_labels, np.array([balanced_label]),np.array([change_number_label]))))
                else:
                    data_list.append(np.concatenate(
                        (np.array([value_function]), irr_data[timestamps].values, mpc_data[timestamps].values,np.array([balanced_label]),np.array([change_number_label]))))




                if label_list is None:
                    # create list of labels
                    label_list = list()  # value function , difference MPC and irradiation data
                    value_function=['VF']
                    irr_l_list = list()
                    mpc_l_list = list()
                    sc_l_list = list()
                    ch_l_list = list()
                    for i in range(len(irr_data[timestamps].values)):
                        irr_l_list.append('IRR' + str(i))  # irradiation data
                        mpc_l_list.append('MPC' + str(i))  # MPC optimal
                        if not regression_data_only:
                            sc_l_list.append('SC' + str(i))  # classes
                            ch_l_list.append('CH' + str(i))  # change_labels

                    label_list = label_list + value_function + irr_l_list + mpc_l_list + sc_l_list + ch_l_list + ['B'] + ['C']  # 'B' for balanced # 'C' for change count
                    print(label_list)

            else:
                print(image_key + " current settings overshoot the end of the available time series")
                break

        # One data frame for a day
        data_df = pd.DataFrame(data=np.array(data_list), index=np.array(index_list), columns=np.array(label_list))

        print(data_df)

        # print in image folder
        if print_to_csv is True:
            path_to_image = os.path.dirname(list(data_tuple[0].values())[0])
            file_name = path_to_image.rsplit('/', 1)[1] + "-labels_small.csv"
            output_path = os.path.join(path_to_image, file_name)
            print("Write to: ", output_path)
            data_df.to_csv(output_path, sep=',')




def create_scaled_classification_data(solar_station=ac.ABB_Solarstation.C, img_d_tup_l=None):
    """
    Creates file with scaled data based on classification. Cloudy is defined as below 70% of clearsky model and is scaled by clearsky_irr/3
    The sunny data points are scaled by  clearsky_irr. This can be used with mpc to produce another optimal policy for classified data, which will be worse than the actual
    optimal mpc. But it can help as a benchmark for a classification problem by comparing the optimal classification mpc with the 
    predicted classification mpc (compare energy throughput per day)
    """
    irr_data_paths = [p[1] for p in abb_rp.read_cld_img_day_range_paths(solar_station=solar_station, suffix='int',
                                                                        img_d_tup_l=img_d_tup_l,
                                                                        randomize_days=False)]
    cs_data_paths = [p[1] for p in abb_rp.read_cld_img_day_range_paths(solar_station=solar_station, suffix='cs',
                                                                       img_d_tup_l=img_d_tup_l,
                                                                       randomize_days=False)]

    for irr_data_p, cs_data_p in zip(irr_data_paths, cs_data_paths):
        irr_data = pd.Series.from_csv(irr_data_p)
        cs_data = pd.Series.from_csv(cs_data_p)

        classes = irr_data < 0.7 * cs_data
        print(classes.between_time("12:00", "13:00"))
        print(cs_data[classes].between_time("12:00", "13:00"), cs_data[classes].between_time("12:00", "13:00") / 3)
        cs_data[classes] = cs_data[classes] / 3

        print(cs_data[classes].between_time("12:00", "13:00"))

        # TODO finish


def energy_throughput_per_day_optimal(solar_station=abb_st.C,
                                      file_filter={"Debevec", ".jpeg"}, print_to_csv=False, path=None, visualize=False):
    """
    Creates per day data of aggregated energy throughput (difference irradiance and  given optimal MPC
    Calculated between automatically inferred time in abb constants
    :return: timeseries day data -> value function (energy throughput)
    """
    date_list = list()
    value_list = list()

    for data_tuple in abb_rp.read_cld_img_time_range_paths(solar_station=solar_station, img_d_tup_l=None,
                                                           img_t_tup_l=None, automatic_daytime=True,
                                                           file_filter=file_filter, get_mpc_data=True,
                                                           randomize_days=False):

        irr_data = pd.Series.from_csv(data_tuple[1])
        mpc_data = pd.Series.from_csv(data_tuple[2])
        diff = pd.Series.abs(irr_data - mpc_data)
        diff = diff.dropna()

        # print(irr_data.index)
        day_string = "%02d" % (int(irr_data.index[0].to_pydatetime().day))
        month_string = "%02d" % (int(irr_data.index[0].to_pydatetime().month))  # extract month with leading zero
        year_string = (str(irr_data.index[0].to_pydatetime().year))

        date = dt.datetime.strptime(year_string + "-" + month_string + "-" + day_string, '%Y-%m-%d')

        if solar_station is abb_st.C:
            t_from, t_to = ac.c_sunrise_sunset[month_string]
            path = os.path.join(ac.c_int_data_path, 'C-optimal-mpc-energy-tp.csv')
        elif solar_station is abb_st.MS:
            path = os.path.join(ac.ms_int_data_path, 'MS-optimal-mpc-energy-tp.csv')
            raise ValueError('Automatic daytime not implemented vor MS yet')
        else:
            raise ValueError('Wrong solar power plant input')

        integral = pd.Series.sum(diff.between_time(t_from.time(), t_to.time()))
        print(date, integral)
        value_list.append(integral)
        date_list.append(date)

    ts = pd.Series(value_list, index=date_list)

    if print_to_csv is True:
        print("Write to: ", path)
        ts.to_csv(path, sep=',')

    if visualize is True:
        ts.plot(x='index')
        plt.show()

    return ts


def images_resize(scale=None, abb_solarstation=ac.ABB_Solarstation.C, img_d_tup_l=None, img_t_tup_l=None,
                  automatic_daytime=False, file_filter={"Debevec", ".jpeg"}):
    if scale is None:
        raise ValueError('Input new scale in Pixels')

    if img_t_tup_l is None:
        automatic_daytime = True

    if img_d_tup_l is None:
        if abb_solarstation is abb_st.C:
            img_d_tup_l = [(ac.c_min_date, ac.c_max_date)]
        elif abb_solarstation is abb_st.MS:
            img_d_tup_l = [(ac.ms_min_date, ac.ms_max_date)]
        else:
            raise ValueError('ABB Solarstation not found (C or MS?)')

    print("Creating list from " + str(img_d_tup_l[0][0]) + ' until ' + str(img_d_tup_l[0][1]))
    """
    image_path_list = [list(data_tuple[0].values()) for data_tuple in abb_rp.read_cld_img_time_range_paths(
        solar_station=abb_solarstation, img_d_tup_l=img_d_tup_l, img_t_tup_l=img_t_tup_l,
        automatic_daytime=automatic_daytime, file_filter=file_filter, randomize_days=False)][0]
    print(len(image_path_list))
    print("Done")
    """
    for day in abb_rp.read_cld_img_time_range_paths(solar_station=abb_solarstation, img_d_tup_l=img_d_tup_l,
                                                    img_t_tup_l=img_t_tup_l, automatic_daytime=automatic_daytime,
                                                    file_filter=file_filter, randomize_days=False):
        image_path_list = list(day[0].values())

        for image_path in image_path_list:
            print(image_path)
            new_path = image_path.rsplit('_', 1)
            output_path = new_path[0] + '_Resize' + str(scale) + '.jpeg'
            print(output_path)

            o_path = pathlib.Path(output_path)

            if not o_path.is_file():

                image = skio.imread(image_path)

                image = skt.resize(image, (scale, scale))

                skio.imsave(output_path, image)

            else:
                print(output_path, "already exists")


def images_draw_sunspots(scale=None, abb_solarstation=ac.ABB_Solarstation.C, img_d_tup_l=None, img_t_tup_l=None,
                  automatic_daytime=False, file_filter={"Debevec", ".jpeg"}):
    if scale is None:
        raise ValueError('Input new scale in Pixels')

    if img_t_tup_l is None:
        automatic_daytime = True

    if img_d_tup_l is None:
        if abb_solarstation is abb_st.C:
            img_d_tup_l = [(ac.c_min_date, ac.c_max_date)]
        elif abb_solarstation is abb_st.MS:
            img_d_tup_l = [(ac.ms_min_date, ac.ms_max_date)]
        else:
            raise ValueError('ABB Solarstation not found (C or MS?)')

    print("Creating list from " + str(img_d_tup_l[0][0]) + ' until ' + str(img_d_tup_l[0][1]))

    for day in abb_rp.read_cld_img_time_range_paths(solar_station=abb_solarstation, img_d_tup_l=img_d_tup_l,
                                                    img_t_tup_l=img_t_tup_l, automatic_daytime=automatic_daytime,
                                                    file_filter=file_filter, get_sp_data=True,randomize_days=False):


        image_keys = list(day[0].keys())


        for key in image_keys:
            image_path = day[0][key]
            print(image_path)


            sunspot_data = pd.read_csv(day[2], index_col=0,parse_dates=True,header=None) # read sp file data with sunspot coordinates

            sunspot_coords = sunspot_data.loc[key].ix[:,0:2].values[0]



            sunspot_x = int(sunspot_coords[0])
            sunspot_y = int(sunspot_coords[1])


            new_path = image_path.rsplit('_', 1)
            output_path = new_path[0] + '_Resize_sp_' + str(scale) + '.jpeg'
            print(output_path)

            o_path = pathlib.Path(output_path)

            if not o_path.is_file():

                image = misc.imread(image_path)

                for i in range(-20,21):
                    for j in range(-20,21):
                        image[sunspot_x + i, sunspot_y + j, 0] = 0.0
                        image[sunspot_x+i,sunspot_y+j,1] = 255.0
                        image[sunspot_x + i, sunspot_y + j, 2] = 0.0

                print(image.shape)
                image = skt.resize(image, (scale, scale))


                #misc.imshow(image)



                skio.imsave(output_path, image)

            else:
                print(output_path, "already exists")




def change_information_histogram(solar_station=ac.ABB_Solarstation.C, bins='auto',density=True):
    if solar_station ==ac.ABB_Solarstation.C:
     suffix='C-'
    elif solar_station ==ac.ABB_Solarstation.MS:
     suffix='MS-'
    else:
        raise ValueError("Illegal solar station")


    irr_data = abb_rp.read_full_int_irr_data(write_to_csv=False)


    day_list = abb_rp.read_cld_img_day_range_paths(solar_station=solar_station)

    diff_list = list()
    path=None

    for days in day_list:
        date = os.path.basename(days[0]).split('-',1)[1]
        irr_data_tmp = irr_data.loc[str(date):str(date)]
        irr_data_tmp = irr_data_tmp.asfreq(freq='10S')
        diff = -(irr_data_tmp.iloc[0:-2].values-irr_data_tmp.iloc[1:-1].values)

        diff_list.append(diff)


        #print(np.argwhere(np.isnan(diff)))
        print("create histogram for "+str(date)+":")

        """
        hist,be = np.histogram(diff,bins='auto',density=True)


        
        plt.hist(diff, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()
        plt.close()
        """
        path = os.path.dirname(days[1])

        #print(os.path.join(path,suffix+date+"_change_hist.csv"))
        #np.savetxt(os.path.join(path,suffix+date+"_change_hist.csv"), np.array([hist, be]))

    total_diff = np.concatenate(diff_list)

    mean_t = np.mean(total_diff)
    std_t = np.std(total_diff)
    median_t = np.median(total_diff)

    print(mean_t,std_t,median_t)

    hist_t, be_t = np.histogram(total_diff, bins='auto', density=False)


    plt.hist(total_diff, bins='auto')  # arguments are passed to np.histogram
    plt.hist(total_diff, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    plt.close()

    np.savetxt(os.path.join(path, suffix + "total_change_hist.csv"), np.concatenate([np.array([mean_t,std_t,median_t]),hist_t, be_t]))




def naive_battery_throughput_calculation(solar_station=ac.ABB_Solarstation.C,change_constraint_wh_min=ac.solar_irradiance / 25):
    """
    Calculate naive, reactive control of battery
    :param solar_station: 
    :param change_constraint_wh_min: 
    :return: 
    """
    day_list = abb_rp.read_cld_img_day_range_paths(solar_station=solar_station,suffix='int')


    for day in day_list:

        naive_control_values=list()
        naive_control_values2=list()

        index_list = list()

        int_data_path = day[1]
        int_data_pd = pd.Series.from_csv(int_data_path)

        #print(int_data_pd.iloc[::4])

        it1 = int_data_pd.iteritems()
        it2 = int_data_pd.iteritems()
        next(it2)

        max_abs_change = change_constraint_wh_min/60

        max_abs_change2 = max_abs_change/2.5

        print(max_abs_change)

        naive_control_values.append(int_data_pd.iloc[0])
        naive_control_values2.append(int_data_pd.iloc[0])


        index_list.append(int_data_pd.index[0])

        for row1,row2 in zip(it1,it2):



            diff = row1[1]-naive_control_values[-1]
            diff2 = row2[1]-naive_control_values2[-1]

            if abs(diff) > max_abs_change:

                next_control = naive_control_values[-1]+ np.sign(diff)*max_abs_change
                naive_control_values.append(next_control)
            else:
                naive_control_values.append(row1[1])

            """
            if abs(diff2) > max_abs_change2:

                next_control = naive_control_values2[-1]+ np.sign(diff2)*max_abs_change2
                naive_control_values2.append(next_control)
            else:
                naive_control_values2.append(row2[1])
            """

            index_list.append(row2[0])




        naive_pd = pd.Series(data=naive_control_values,index=index_list)
        #naive_pd2 = pd.DataFrame(data=naive_control_values2, index=index_list)

        output_path = int_data_path.rsplit('-',1)[0]+'-naive'+str(int(change_constraint_wh_min))+'_new.csv'


        pd.Series.to_csv(naive_pd,path=output_path)
        #output_path=
        """
        plot_pd = pd.concat([pd.Series.to_frame(naive_pd),pd.Series.to_frame(int_data_pd)],axis=1)

        print(naive_pd,int_data_pd)

        plot_pd.plot()
        plt.show()
        """

def create_rl_environment_input_files(solar_sation = ac.ABB_Solarstation.C, automatic_daytime=True,file_filter={"Debevec", ".jpeg"},output_path=None,output_name="rl_data.csv"):
    """
    Creates dataframe used by reinforcement learning algorithms
    :param solar_sation:
    :param automatic_daytime:
    :param file_filter:
    :param output_path:
    :return:
    """

    day_list = list()

    for day in abb_rp.read_cld_img_time_range_paths(solar_station=solar_sation,automatic_daytime=automatic_daytime,file_filter=file_filter,get_mpc_data=True,get_cs_data=True):

        print(day)

        full_irr_data = pd.read_csv(day[1],header=None,index_col=0,parse_dates=True).sort_index()
        full_mpc_data = pd.read_csv(day[2], header=None, index_col=0, parse_dates=True).sort_index()
        full_cs_data = pd.read_csv(day[3], header=None, index_col=0, parse_dates=True).sort_index()

        data_df = pd.concat([full_irr_data,full_mpc_data,full_cs_data],axis=1)

        day_keys = list(day[0].keys())

        day_rl_pd= data_df.loc[pd.DatetimeIndex(day_keys)].sort_index()

        if day_rl_pd.isnull().values.any():
            print(day[0])
        day_rl_pd.columns=['irr','mpc','cs']
        day_paths = [os.path.join(*pathlib.Path(day[0][k]).parts[-2:]) for k in day_keys] #img name and parent folder to allow file location flexibility
        day_rl_pd['img_name'] = day_paths

        day_list.append(day_rl_pd)




    rl_pd = pd.concat(day_list,axis=0).sort_index()

    if output_path is not None:
        rl_pd.to_csv(os.path.join(output_path,output_name))



    return rl_pd




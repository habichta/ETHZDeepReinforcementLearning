'''
Contains functions to simplify reading of the input data. This can be illuminance data or cloud images
Author: Arthur Habicht habichta@ethz.ch
'''

from . import abb_clouddrl_constants as ac
from .abb_clouddrl_constants import ABB_Solarstation
from .abb_clouddrl_constants import abb_filepattern
from . import abb_clouddrl_transformation_pipeline
from . import abb_clouddrl_visualization_pipeline
import itertools as it
import datetime as dt
import os
import random
from collections import OrderedDict
import pandas as pd

img_path_dict = {ABB_Solarstation.C: ac.c_img_path,
                 ABB_Solarstation.MS: ac.ms_img_path}

data_path_dict = {ABB_Solarstation.C: ac.c_int_data_path,
                  ABB_Solarstation.MS: ac.ms_int_data_path}


def read_cld_img(solar_station=ABB_Solarstation.C, images_l=None):
    """
    Input: specific images or days (depending on time input format)
    images is list of datetime format i.e. '2014-02-07 11:52:21' or '2014-02-07' for whole day
    Read next existing image: f.e. if time is 11:52:21 read image at that time or chronologically next image
    If images = none, read all images
    """

    pass


def read_cld_img_day_range_paths(solar_station=ABB_Solarstation.C, suffix='int',img_d_tup_l=None, randomize_days=False):
    """
    Returns a tuple of path to image and path to interpolated data for the days that are within a list of time ranges
    img_d_tup_l=[(day_from, day_to),()...] where day_from and day_to are datetime objects (use strptime). The ranges
    are not allowed to overlap. Choose which solar station and whether the output should be sorted (default) by date or
    randomized, retruns tuple (path to images,path to file depending on suffix)
    """

    if img_d_tup_l is None:
        if solar_station is ABB_Solarstation.C:
            img_d_tup_l = [(ac.c_min_date,ac.c_max_date)]
        elif solar_station is ABB_Solarstation.MS:
            img_d_tup_l = [(ac.ms_min_date, ac.ms_max_date)]


    img_path = img_path_dict[solar_station]
    data_path = data_path_dict[solar_station]

    range_list = img_d_tup_l

    # index [range][to/from]
    # sort ranges according to beginning day
    range_list.sort(key=lambda e: e[0])

    if any([d[0][1] >= d[1][0] for d in zip(range_list[:-1], range_list[1:])]):
        raise ValueError('Overlapping date ranges!')
    if any([d[0] > d[1] for d in range_list]):
        raise ValueError('End of day date range is before day date beginning of range! Check input!')

    valid_day_paths = []
    day_files = sorted([f for f in next(os.walk(img_path))[1]])

    for d_file in day_files:
        d_file_date = d_file.split('-', 1)

        try:
            date_obj = dt.datetime.strptime(d_file_date[1], '%Y-%m-%d')
        except Exception as e:
            print('There were files that could not be read:', d_file_date)

        if any(list(map(lambda r_tup: r_tup[0] <= date_obj <= r_tup[1], range_list))):
            valid_day_paths.append(
                (os.path.join(img_path, d_file), os.path.join(data_path, d_file + '-'+suffix+'.csv')))

    # valid_day_paths contains valid paths to days within ranges (sorted).

    return random.sample(valid_day_paths, len(valid_day_paths)) if randomize_days else sorted(valid_day_paths)  #valid_day_paths is [(img_path,path to data given suffix),..]


def read_full_int_irr_data(solar_station=ABB_Solarstation.C,write_to_csv=False):

    if solar_station == ABB_Solarstation.C:
        file_name = "C-full-int.csv"

    elif solar_station == ABB_Solarstation.MS:
        file_name = "MS-full-int.csv"
    else:
        raise ValueError("Illegal station")
    full_path = os.path.join(ac.c_int_data_path,file_name)

    if os.path.isfile(full_path):
       int_pd = pd.Series.from_csv(full_path)
       return int_pd


    file_paths = read_cld_img_day_range_paths(solar_station)
    int_file_paths = [path[1] for path in file_paths]
    dataframe_list = list()

    for p in int_file_paths:
        print(p)
        dataframe_list.append(pd.Series.from_csv(p))

    int_pd = pd.concat(dataframe_list,axis=0)

    if write_to_csv:
        pd.Series.to_csv(int_pd,path=full_path)

    return int_pd




def image_key_creator(image_string):
    path_parts = image_string.split('_')
    return ' '.join(('-'.join(path_parts[0:3]), ':'.join(path_parts[3:6])))


def read_cld_img_time_range_paths(solar_station=ABB_Solarstation.C, img_d_tup_l=None, automatic_daytime=False,automatic_daytime_threshold = 0.2,
                                  img_t_tup_l=None, file_filter={"Debevec", ".jpeg"}, randomize_days=False,
                                  get_mpc_data=False, get_cs_data=False,get_sp_data=False, randomize_time_batch=0):
    """
    Returns a tuple that contains an ordered dictionary (according of time of say) the key looks like: Year-Month-Day Hour:Minute:Second and maps to the corresponding
    path to the image as a value. The second element of the tuple is the path to the interpolated data file of that particular day. This is a generator, so
    the function has to be called in a loop. It generates the paths to all images within the time ranges in img_t_tup_l and filters out all files that do not contain the
    strings in file_filter. The output will have the same order as the date of the days (default:ordered, can be randomized). For now, the dictionary containing the times
    only outputs the times in sorted fashion (randomize_time_batch not implemented yet)
    automatic_daytime = True: Uses the clear sky model to caluclate good starting and ending times. Only consider data between time_fomr and time_to calcualted using the
    automatic daytime threshold. Z.b. 0.2 means that only consider times where irradiance in the clear sky model is larger than 0.2 of the maximum of the clear sky model at
    that particular day.
    Output: (dict(time->image path), path to data file,path to mpc,path to sp)
    """



    day_path_list = read_cld_img_day_range_paths(
        solar_station=solar_station, img_d_tup_l=img_d_tup_l, randomize_days=randomize_days)

    if img_t_tup_l is not None:
        if any([d[0][1] >= d[1][0] for d in zip(img_t_tup_l[:-1], img_t_tup_l[1:])]):
            raise ValueError('Overlapping time ranges!')
        if any([d[0] > d[1] for d in img_t_tup_l]):
            raise ValueError('End of time range is before beginning of time range! Check input!')
    for day_path in day_path_list:

        """
        if automatic_daytime:
            month_string = os.path.basename(day_path[0]).split('-')[
                2]  # extract month value from z.b. C-2016-03-23 -> 03
            if solar_station is ABB_Solarstation.C:
                t_from, t_to = ac.c_sunrise_sunset[month_string]
                img_t_tup_l = [(t_from, t_to)]
            elif solar_station is ABB_Solarstation.MS:
                raise ValueError('Automatic daytime not implemented vor MS yet')
            else:
                raise ValueError('Wrong solar power plant input')
        """
        if automatic_daytime:
            cs_path = day_path[1].rsplit('-', 1)[0] + '-cs.csv'
            cs_data = pd.Series.from_csv(cs_path)
            threshold_irradiance = cs_data.max() * automatic_daytime_threshold
            cs_ind = cs_data[cs_data>=threshold_irradiance].index
            t_from = cs_ind[0]
            t_to = cs_ind[-1]
            img_t_tup_l = [(t_from,t_to)]
            print(str(day_path[1]) + " restricted to automatic daytimes: ", img_t_tup_l)





        if img_t_tup_l is not None:
            sorted_image_path_dict = OrderedDict(
                {image_key_creator(t): os.path.join(day_path[0], t) for t in sorted(os.listdir(day_path[0])) if
                 all(filter_word in t for filter_word in file_filter)
                 and any(list(map(lambda r_tup: r_tup[0].time() <= dt.datetime.strptime(':'.join(t.split('_')[3:6]),
                                                                                        '%H:%M:%S').time() <= r_tup[
                                                    1].time(), img_t_tup_l)))})
        else:
            sorted_image_path_dict = OrderedDict(
                {image_key_creator(t): os.path.join(day_path[0], t) for t in sorted(os.listdir(day_path[0])) if
                 all(filter_word in t for filter_word in file_filter)})


        data_list = list()

        data_list.extend([sorted_image_path_dict,day_path[1]])


        #TODO: Nr. of different files grew, create pandas DF with all data ...

        if solar_station is not ABB_Solarstation.C:
            raise ValueError('mpc, cs not yet available for this solar powerplant')

        if get_mpc_data:
            mpc_path = day_path[1].rsplit('-', 1)[0] + '-mpc.csv'
            data_list.append(mpc_path)

        if get_cs_data:
            cs_path = day_path[1].rsplit('-', 1)[0] + '-cs.csv'
            data_list.append(cs_path)

        if get_sp_data:
            cs_path = day_path[1].rsplit('-', 1)[0] + '-sp.csv'
            data_list.append(cs_path)
        """
        if not get_mpc_data:
            yield (sorted_image_path_dict, day_path[
                1])  # returns a tuple that contains a dictonary with all image paths in one day and as a second element the path to the interpolated data)
        else:

            mpc_path = day_path[1].rsplit('-', 1)[0] + '-mpc.csv'
            yield (sorted_image_path_dict, day_path[1], mpc_path)
        """

        data_tuple = tuple(data_list)

        yield data_tuple




def read_cld_img_index():
    pass


def create_cld_img_index():
    pass

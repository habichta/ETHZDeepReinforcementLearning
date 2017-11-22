

from enum import Enum
import re
import datetime as dt


class ABB_Solarstation(Enum):
    C = 1
    MS = 2

path = '/media/data/Daten'

c_img_path = '/media/data/Daten/img_C'
c_int_data_path = '/media/data/Daten/data_C_int'

ms_img_path = '/media/data/Daten/img_MS'
ms_int_data_path = '/media/data/Daten/data_MS_int'

abb_ms_time_pattern = re.compile("(.[0-9]{2}\,[0-9]{2}\,[0-9]{2}.)")
abb_c_time_pattern = re.compile("(.[0-9]{2}\:[0-9]{2}\:[0-9]{2}.)")
abb_filepattern = re.compile("(.-[0-9]{4}\-[0-9]{2}\-[0-9]{2}.log)")

solar_irradiance = 1000  # W/m^2

# Month to time
"""
WARNING: Do not use these dates: use the clear sky model instead. This is already implemented in  read_cld_img_time_range_paths
use this function instead.
"""
c_sunrise_sunset = {'07': (dt.datetime.strptime('06:00:00', '%H:%M:%S'), dt.datetime.strptime('20:30:00', '%H:%M:%S')),
                    '08': (dt.datetime.strptime('06:30:00', '%H:%M:%S'), dt.datetime.strptime('20:00:00', '%H:%M:%S')),
                    '09': (dt.datetime.strptime('07:00:00', '%H:%M:%S'), dt.datetime.strptime('19:00:00', '%H:%M:%S')),
                    '10': (dt.datetime.strptime('07:00:00', '%H:%M:%S'), dt.datetime.strptime('18:00:00', '%H:%M:%S')),
                    '11': (dt.datetime.strptime('07:30:00', '%H:%M:%S'), dt.datetime.strptime('17:00:00', '%H:%M:%S')),
                    '12': (dt.datetime.strptime('07:30:00', '%H:%M:%S'), dt.datetime.strptime('16:30:00', '%H:%M:%S')),
                    '01': (dt.datetime.strptime('07:40:00', '%H:%M:%S'), dt.datetime.strptime('17:00:00', '%H:%M:%S')),
                    '02': (dt.datetime.strptime('07:20:00', '%H:%M:%S'), dt.datetime.strptime('17:30:00', '%H:%M:%S')),
                    '03': (dt.datetime.strptime('06:30:00', '%H:%M:%S'), dt.datetime.strptime('18:30:00', '%H:%M:%S')),
                    '04': (dt.datetime.strptime('06:20:00', '%H:%M:%S'), dt.datetime.strptime('20:00:00', '%H:%M:%S'))}

c_min_date = dt.datetime.strptime('2015-07-16', '%Y-%m-%d') #2015-07-15 is corrupted
c_max_date = dt.datetime.strptime('2016-04-25', '%Y-%m-%d')

c_data_date_start = dt.datetime.strptime('2015-07-15', '%Y-%m-%d')
c_data_date_end = dt.datetime.strptime('2016-04-25', '%Y-%m-%d')


ms_min_date = dt.datetime.strptime('2015-07-05', '%Y-%m-%d')
ms_max_date = dt.datetime.strptime('2016-04-21', '%Y-%m-%d')

ms_data_date_start = dt.datetime.strptime('2015-07-05', '%Y-%m-%d')
ms_data_date_end = dt.datetime.strptime('2016-04-21', '%Y-%m-%d')

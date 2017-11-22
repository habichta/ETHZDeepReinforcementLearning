from pvlib import clearsky
from pvlib.location import Location
from pvlib import solarposition
from pvlib import atmosphere
from . import abb_clouddrl_constants as ac
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt
import os




def ineichen_series(path_c=None,path_ms=None,print_to_csv=False,visualize=False):


        cav_tz = timezone('Europe/Rome')
        ms_tz = timezone('Europe/Zurich')

        cavriglia_loc = Location(latitude=43.5354,longitude=11.4814,altitude=308,tz=cav_tz)
        montsoleil_loc = Location(latitude=47.1642,longitude=6.9903,altitude=1278,tz=ms_tz)

        print("create date_range")
        dates_c = pd.date_range(start=ac.c_data_date_start, end=ac.c_data_date_end, freq="D")
        dates_ms = pd.date_range(start=ac.ms_data_date_start, end=ac.ms_data_date_end, freq="D")


        for day in dates_c:
            try:
                print(day.date())

                cs_path = os.path.join(ac.c_int_data_path, "C-"+str(day.date())+"-int.csv")

                print(cs_path)

                int_data = pd.Series.from_csv(
                    path=cs_path, sep=',', index_col=0, infer_datetime_format=True)

                times = int_data.index

                #times = pd.date_range(start=str(day)+" "+"05:50:00",end=str(day)+" "+"20:45:00",freq="240S")

                print("time localize")
                times_localized= times.tz_localize(cavriglia_loc.tz,ambiguous=True)
                print("calculate athmosphere parameters and sun position")
                ephem_data= solarposition.get_solarposition(times_localized,cavriglia_loc.latitude,cavriglia_loc.longitude)
                am= atmosphere.relativeairmass(ephem_data['apparent_zenith'])
                am= atmosphere.absoluteairmass(am, atmosphere.alt2pres(cavriglia_loc.altitude))

                print("calculate clearsky..")
                out_c_int = clearsky.ineichen(ephem_data['apparent_zenith'],am,3)['ghi']


                #remove timezone info to make later processes easier, avoid automatic conversion to UTC


                out_c_int.index = out_c_int.index.tz_localize(None)
                out_c_int = out_c_int.astype(float)


                #print(out_c.index)
                print(out_c_int)
                # print(out_c_int.loc['2015-07-15 20:29:55'])

                if print_to_csv is True:
                    name = "C-"+str(day.date()) + "-cs.csv"
                    path = os.path.join(path_c, name)
                    print("Write to: ", path)
                    out_c_int.to_csv(path, sep=',', index=True)

                if visualize:
                    out_c_int.plot(x=None)
                    plt.show()

            except Exception as e:
                print(e)









from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_rp
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import matplotlib as plt
import pandas as pd
import os
import matplotlib.pyplot as plt




path = os.path.join(abb_c.c_int_data_path,"C-throughput.csv")

diff_df = pd.DataFrame.from_csv(path)

diff_t_df = diff_df[['mpc100','naive100','mpc40','naive40']]

diff100 = diff_df[['mpc100','naive100']]

diff40 = diff_df[['mpc40','naive40']]


sum_diff = diff_t_df.sum()
sum_diff100 = diff100.sum()
sum_diff40 = diff40.sum()


sum_diff.plot(style='o')
plt.show()

diff_t_df.plot()
plt.show()


print(sum_diff)



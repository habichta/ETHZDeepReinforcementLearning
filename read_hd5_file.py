import pandas as pd

path="/media/data/Daten/data_C_int/C-data-master.h5"

df = pd.read_hdf(path)

print(df.head(2))

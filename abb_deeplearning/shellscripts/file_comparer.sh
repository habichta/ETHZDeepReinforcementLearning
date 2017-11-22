#compare image and data contensts. check which days are missing either in data or image folders
# 3 columns: 1. only in data 2. only in images 3. exists in both

C_IMG_DAYS="/media/data/Daten/c_img_days.log"
C_DATA_DAYS="/media/data/Daten/c_data_days.log"
MS_IMG_DAYS="/media/data/Daten/ms_img_days.log"
MS_DATA_DAYS="/media/data/Daten/ms_data_days.log"


cd /media/data/Daten/data_C
ls | sed -e s/[^0-9]//g > c_data_days.log
mv c_data_days.log ..

cd /media/data/Daten/img_C
ls | sed -e s/[^0-9]//g > c_img_days.log
mv c_img_days.log ..


cd /media/data/Daten/data_MS
ls | sed -e s/[^0-9]//g > ms_data_days.log
mv ms_data_days.log ..


cd /media/data/Daten/img_MS
ls | sed -e s/[^0-9]//g > ms_img_days.log
mv ms_img_days.log ..



comm  <(cat $C_DATA_DAYS|sort) <(cat $C_IMG_DAYS|sort) > "/media/data/Daten/c_diff.log"

comm  <(cat $MS_DATA_DAYS|sort) <(cat $MS_IMG_DAYS|sort) > "/media/data/Daten/ms_diff.log"

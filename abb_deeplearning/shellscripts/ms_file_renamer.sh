#MS_DATA

#Rename MS Files for uniform naming scheme: MS-Date.log
cd /media/data/Daten/data_MS
for file in *
do
	echo $file
	mv "$file" "MS-${file/.txt/.log}"
done



#C_DATA

#Rename C Files for uniform naming scheme: C-Date.log
#disk mounted: sudo mount /dev/sdb1 /media/data

#Extract files first, then move, install: sudo apt-get install p7zip  to extract 7z files

C_DATA_PATH="/media/data/Daten/data_C"

cd /media/data/Daten/HDRImages_Cavriglia/img
for file in *
do
	extract
	for f in $file/*_data.7z
	do

		echo "extracting $f ..."
		7z x "$f"

	done

done

#moving

for file in *.log
do
	echo "moving $file to $C_DATA_PATH/C-${file/_data.log/.log}"
	mv "$file" "$C_DATA_PATH/C-${file/_data.log/.log}"
done
	

cd /media/data/Daten/Data_C

for file in *
do
	echo "moving $file to ${file//_/-}"
	mv "$file" "${file//_/-}"
done
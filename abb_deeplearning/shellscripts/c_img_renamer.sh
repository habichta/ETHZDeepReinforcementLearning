#move C images to folder if they contain jpegs
#Ony move if more than 1000 pictures, other samples are
# seen as corrupted
cd /media/data/Daten/HDRImages_Cavriglia/img

TARGET_PATH="/media/data/Daten/img_C"

for file in *
do

	if [[ $(find $file -type f | wc -l) -gt 1000 ]]
		then
		echo "$(find $file -type f | wc -l) in $file"

			mv $file $TARGET_PATH

	fi

done


cd /media/data/Daten/img_C

for file in *
do
	echo "moving $file to C-${file//_/-}"
	mv "$file" "C-${file//_/-}"
done
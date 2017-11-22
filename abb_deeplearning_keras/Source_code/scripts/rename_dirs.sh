#!/usr/bin/env bash

# Script loop through all the sub-directories, files for a given root directory.
# Cleans up messy file extensions like (eg: *_7z_) by renaming them with clean ones.

#Usage:
#command line arguments - {source folder}

#folder=/home/maverick/Desktop/temp/TS_Data
folder=$1

#Pass the root folder name here
for directory in $(find $folder -type d);
do
    echo $directory
    #cd $directory
    for file in $directory/*;
    do
        echo $file ">"  "${file//_}"
        # mv $file ${file//_}
        if [[ $file == *_txt_* ]]
            then
                mv "$file" "${file%._txt_}.txt"
            elif [[ $file == *mat_* ]]; then
                    mv "$file" "${file%._mat_}.mat"
                elif [[ $file == *tar_* ]]; then
                        mv "$file" "${file%._tar_}.tar"
                    elif [[ $file == *7z_* ]]; then
                            mv "$file" "${file%._7z_}.7z"

        fi
    done
done


# for file in *;
# do
#     echo $file ">"  "${file//_}"
#     # mv $file ${file//_}
#     if [[ $file == *txt_* ]]
#         then
#             mv "$file" "${file%._txt_}.txt"
#         elif [[ $file == *mat* ]]; then
#                 mv "$file" "${file%._mat_}.mat"
#             elif [[ $file == *tar_* ]]; then
#                     mv "$file" "${file%._tar_}.tar"
#                 elif [[ $file == *7z_* ]]; then
#                         mv "$file" "${file%._7z_}.7z"

#     fi
# done
 
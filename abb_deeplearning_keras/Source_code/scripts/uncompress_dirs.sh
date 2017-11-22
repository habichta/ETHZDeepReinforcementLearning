#!/usr/bin/env bash

# Script loops through all the .tar.gz archives in the input folder, 
# and extracts each archive in a seperate output folder

#Usage:
#command line arguments - {source folder} {destination folder}



source_dir=$1;
destination_dir=$2;


cd $source_dir



for dir in $source_dir*.gz
do
	echo $dir
	tar -xzvf "$dir" -C $destination_dir
done

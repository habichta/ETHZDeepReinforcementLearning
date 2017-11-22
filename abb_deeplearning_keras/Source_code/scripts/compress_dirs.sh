#!/usr/bin/env bash

# Script loops through all the sub-directories, and creates compressed archives for each folder

#Usage:
#command line arguments - {source folder} {destination folder}



source_dir=$1;
destination_dir=$2;


cd $source_dir

for dir in */
do
	echo $dir
	base=$(basename "$dir")
	tar -czf "$destination_dir${base}.tar.gz" "$dir"
done

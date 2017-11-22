#!/bin/bash

path=/local/habichta/keras_experiment/
code=/home/habichta/dlabb/abb_deeplearning_keras/Source_code/models/


for e in balanced_16_224_pertrained_with_irradiance_31
do
	
	fullpath=${path}${e}/eval_settings
	for setting in "$fullpath"/*
	do
		echo ${setting}
		KERAS_BACKEND=tensorflow python ${code}predict_helper2.py $setting

	done
	
done

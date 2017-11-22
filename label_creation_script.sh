#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/

python ${path}label_creator.py

img_per_sample=2
strides=6
suffix=256
image_name=Resize256

python ${path}nn_tfrecords_options.py $suffix $img_per_sample $strides $image_name
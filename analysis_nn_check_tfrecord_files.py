import tensorflow as tf
import abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants as ac
import os

day = "C-2015-10-10"
day_path = os.path.join(ac.c_img_path,day)
img_nr = 2
strides = 6
suffix = 256


tf_name = os.path.basename(day_path) + '-paths' + "I" + str(img_nr) + "S" + str(strides) + str(
                suffix) + ".tfrecords"

tf_path = os.path.join(ac.c_img_path,day,tf_name)

print(tf_path)

for lines in tf.python_io.tf_record_iterator(tf_path):
    print(lines)
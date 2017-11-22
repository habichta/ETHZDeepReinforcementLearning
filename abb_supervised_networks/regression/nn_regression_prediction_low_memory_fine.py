import sys

sys.path.append('/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/')
sys.path.append('/home/dladmin/Documents/arthurma/shared/dlabb')

import csv
import datetime as dt
import math
import os
import random
import time
import pandas as pd

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
from abb_deeplearning.abb_neuralnet_helpers.nn_sv_input_pipeline_class_low_memory_train_test import ABBTFInputPipeline

# from  models.slim.datasets import dataset_factory
import net_resnetV2
from tensorflow.python.client import device_lib
# from  models.slim.preprocessing import preprocessing_factory

from abb_deeplearning.abb_neuralnet_helpers import custom_preprocessing
from net_input_layers import input_layer_factory
from net_output_layers import output_layer_factory
from net_network_layers import network_factory
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline import image_key_creator
from shutil import copyfile
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac


"""
Adapted ResNet V1 50 Layer with custom input pipeline and changed output/loss layer. 
Use Pretrained model to reduce training time for new task.

"""

FLAGS = tf.app.flags.FLAGS

#######################
# Description#
#######################

tf.app.flags.DEFINE_string(
    'description', 'Resnet still in testing phase',
    'description of experiment will be printed to csv file')

#######################
# Labels #
#######################

irr_label_nr = 31

value_label = ["VF"]
irr_label = ["IRR"+str(i) for i in range(irr_label_nr)] #IRR0 : current, IRR1 current+20s ,IRR30: 10 minutes
mpc_label = ["MPC"+str(i) for i in range(irr_label_nr)]
sc_label = ["SC"+str(i) for i in range(irr_label_nr)]
ch_label = ["CH"+str(i) for i in range(irr_label_nr)]
b_label = ["B"]
c_label = ["C"]

label_key_list = value_label+irr_label+mpc_label+sc_label+ch_label+b_label+c_label





print(label_key_list)

#######################
# Dataset and Input, Directories #
#######################

tf.app.flags.DEFINE_boolean(
    'overwrite_existing_dir', False,
    'Overwrite existing directories for event logs')

tf.app.flags.DEFINE_string(
    'cpk_dir', '/home/dladmin/Documents/arthurma/experiments/simple_resnetv218/regression_resnetV218_bottleneck_linear11_in-2_ih-84_bs-32_kp-0.5_bal-False_is-False_lf-abs_maxw-0.5/-44646',
    'Directory where checkpoints are')


tf.app.flags.DEFINE_string(
    'eval_dir', '/home/dladmin/Documents/arthurma/runs/default',
    'Directory where event logs are written to.')


tf.app.flags.DEFINE_bool(
    'run_once', True,
    'Only run once, can be useful to run several times if the training is done in parallel')

tf.app.flags.DEFINE_bool(
    'write_predictions_to_csv', True,
    'Write pandas file of predictions for each sample (potentially huge file!)')



tf.app.flags.DEFINE_integer('eval_example_num', None, """evaluation of average loss over num examples, if None
go over all examples in the test set""")
# eval_example_num/batch_size define nr of iterations in which the loss os caluclated and averaged

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 10,
    'seconds to wait between evaluations')  # for parallel testing and training


tf.app.flags.DEFINE_bool(
    'use_manual_set_selector', True,
    'Create your own test_list.out file with the paths to the tfrecord files. Files should be in same directroy as code.'
    'They will be copied to the corresponding eval folder')

tf.app.flags.DEFINE_integer(
    'print_outlier', 200,
    'print info about test batches that have loss higher/equal to given number. None to switch off')


tf.app.flags.DEFINE_string(
    'loss_function','abs_max' ,
    'abs or abs_max')

tf.app.flags.DEFINE_float(
    'abs_max_weight',0.5 ,
    'How much to weight maximum term in abs_max relative to abs error')


# The following parameters can be used to automatically create data set splits
#deactivated if use_manual_set_selector is true
#############################################################################
tf.app.flags.DEFINE_integer(
    'train_val_test_split_seed', 1,
    'seed that determines the split of test, validation and train set')


tf.app.flags.DEFINE_float(
    'train_set_size', 0.6,
    'size of train_set vs test set. Between 0 and 1')

tf.app.flags.DEFINE_float(
    'validation_set_size', 0.1,
    'size of validation_set. Between 0 and 1')

tf.app.flags.DEFINE_float(
    'test_set_size', 0.3,
    'size of test_set. Between 0 and 1')

###############################################################################

tf.app.flags.DEFINE_integer(
    'image_name_suffix', 256,
    'suffix for tfrecords')

tf.app.flags.DEFINE_integer(
    'image_height', 256,
    'Height of input image')

tf.app.flags.DEFINE_integer(
    'image_width', 256,
    'Width of input image')

tf.app.flags.DEFINE_integer(
    'image_channels',3,
    'Channels of input image')

tf.app.flags.DEFINE_integer(
    'image_height_resize', 128,
    'Height of input image')

tf.app.flags.DEFINE_integer(
    'image_width_resize',128,
    'Width of input image')

tf.app.flags.DEFINE_integer(
    'image_num_per_sample', 2,
    'Images per sample')


tf.app.flags.DEFINE_boolean(
    'difference_images',False,'take difference between consecutive images instead of images themselves, reduces input size by 1'
)

tf.app.flags.DEFINE_string(  # resize images to this size
    'apply_mask_path', '/media/data/Daten/img_C/cavriglia_skymask256.png',
    'apply image mask in preprocessing, None = no mask. Mask is 256x256x3!')

tf.app.flags.DEFINE_boolean(
    'image_channel_clipping',False,'image between 0 and 1'
)
tf.app.flags.DEFINE_boolean(
    'image_standardization',False,'standardize images by mean and variance after all other preprocessing steps'
)
tf.app.flags.DEFINE_boolean(
    'image_subtract_mean',False,'subtract mean over all image, only if standardization is off'
)



tf.app.flags.DEFINE_integer(
    'strides', 6,
    'Time distance between each image')

tf.app.flags.DEFINE_integer(  # keep 2, others not tested
    'stack_axis', 2,
    'Image stack axis')

# TEST QUEUE

tf.app.flags.DEFINE_bool(
    'shuffle_days_in_test_input_queue', False,
    'shuffles the day tfrecords in the test input queue. If this is False, the training will happen ordered by date of day'
    'but may still be random within days')

tf.app.flags.DEFINE_bool(
    'shuffle_batches_in_test_output_queue', False,
    'shuffles batches  in the test output  queue. Will not randomize batches if False. Note: can randomize among days even if,'
    'shuffle_days_in_test_input_queue is set to False! The days will be read in sequentially but there may be overlap between successive dates, leading'
    'to a batch shuffling among days. Shuffling only within days while keeping day order in tact is not possible yet.')

tf.app.flags.DEFINE_integer(
    'min_batches_in_test_queue',5000,
    'shuffles the day tfrecords in the queue')

tf.app.flags.DEFINE_integer(
    'num_of_test_tfrecord_readers', 1,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_test_queuerunners', 1,
    'Number of queue runners that fill output test fifo queue')

tf.app.flags.DEFINE_integer(
    'capacity_batches_in_test_output_queue', 20,
    'capacity of the test output queue that contains raw images and label batches')

tf.app.flags.DEFINE_integer('test_batch_size', 1, """Images in batch validation""")
#######################
# Model Flags #
#######################

tf.app.flags.DEFINE_string(
    'network_architecture', 'regression_resnetV218_bottleneck', 'The name of the architecture to train.')


tf.app.flags.DEFINE_string(
    'input_layer', 'identity', 'input layer selector')

tf.app.flags.DEFINE_string(
    'output_layer', 'linear11_irr', 'input layer selector')

tf.app.flags.DEFINE_integer(
    'prediction_nr', 11, 'which values should be extracted from the labels (11= each minute, 31 = each 20 sec, change output layer accordingly') #resnet needs linear11 or linear11_irr


p_label = ["P"+str(i) for i in range(FLAGS.prediction_nr)]
l_label = ["L"+str(i) for i in range(FLAGS.prediction_nr)]

eval_file_columns = p_label+l_label




tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# RESNETV2 #
#######################

tf.app.flags.DEFINE_float(
    'batch_norm_decay', 0.99, 'decay for moving average estimation')
tf.app.flags.DEFINE_float(
    'batch_norm_epsilon', 1e-5, 'batchnorm epsilon')
tf.app.flags.DEFINE_boolean(
    'batch_norm_scale', True, '')

tf.app.flags.DEFINE_boolean(
    'global_pool', True, '')
tf.app.flags.DEFINE_boolean(
    'spatial_squeeze', True, '')
tf.app.flags.DEFINE_integer(
    'output_stride', None, '')

tf.app.flags.DEFINE_boolean(
    'include_root_block', True, '')  # should be True, otherwise performance hit. Unless custom input layer

tf.app.flags.DEFINE_boolean(
    'irr_pooling_layer', False, 'pooling or additional conviolutional layer for resnet with current irradiation')


#######################
# SIMPLE_DQN #
#######################

tf.app.flags.DEFINE_integer(
    'simple_dqn_outputs', 11, 'number of predictions')

tf.app.flags.DEFINE_float(
    'dqn_weight_decay', 0.00004, 'The weight decay on the model weights.')  # 0.00004
tf.app.flags.DEFINE_float(
    'dqn_keep_prob', 0.5, 'dropout keep prob for dqn_simplenet_do')  # 0.00004

tf.app.flags.DEFINE_float(
    'dqn_batch_norm_decay', 0.99, 'decay for moving average estimation')
tf.app.flags.DEFINE_float(
    'dqn_batch_norm_epsilon', 1e-5, 'batchnorm epsilon')
tf.app.flags.DEFINE_boolean(
    'dqn_batch_norm_scale', True, '')

#######################
# Image Preprocessor #
#######################


def custom_preprocessor(image):
    # TODO: add seed to preprocessor to apply same random action to all images in a sample

    # image is between 0.0 and 255.0

    # needs to start with reshaping to original size!
    # Only nearest_neighbor worked as resize method

    # image = tf.image.adjust_brightness(image,0.1)


    with tf.variable_scope('convert_image_type') as scope:
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # image = tf.reshape(image, ( FLAGS.image_height_resize, FLAGS.image_width_resize,  FLAGS.image))
    if FLAGS.apply_mask_path:  # TODO: inefficient to repeatedly load mask
        with tf.variable_scope('apply_image_mask') as scope:
            mask = tf.read_file(FLAGS.apply_mask_path)
            mask = tf.image.decode_png(mask)
            if mask.dtype != tf.float32:
                mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
            image = tf.multiply(image, mask)

    with tf.variable_scope('image_resize') as scope:
        image = tf.image.resize_images(image, [FLAGS.image_height_resize, FLAGS.image_width_resize],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    """
    image = custom_preprocessing.preprocess_image(image=image, output_height=FLAGS.image_height_resize,
                                                  output_width=FLAGS.image_width_resize,
                                                   is_training=FLAGS.is_training)
    """

    if FLAGS.image_channels==1:
        with tf.variable_scope('image_to_greyscale') as scope:
            image = tf.image.rgb_to_grayscale(image)

    if FLAGS.image_channel_clipping:
        with tf.variable_scope('image_clipping') as scope:
            image = image / 255.0

    if FLAGS.image_standardization:
        with tf.variable_scope('image_standardization') as scope:
            image = tf.image.per_image_standardization(image)

    if not FLAGS.image_standardization and FLAGS.image_subtract_mean:
        with tf.variable_scope('image_subtract_mean') as scope:
            image = image - tf.reduce_mean(image)



    # float_image.set_shape([height, width, 3])
    # print(image.get_shape())
    # image = tf.image.central_crop(image,0.8)
    # image = tf.random_crop(image, [100,100, 3])
    # image = tf.image.random_flip_left_right(image)
    # print(image.get_shape())
    return image

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def get_nr_of_samples_in_sets(test_data_paths):
    tr = 0
    for fn in test_data_paths:
        for _ in tf.python_io.tf_record_iterator(fn):
            tr += 1

    print("Samples in train per epoch: ", tr)
    print("Steps/Batches in test epoch: ", tr / FLAGS.test_batch_size)
    return tr


def _convert_to_tfrecord_paths(data_list,solar_station=ac.ABB_Solarstation.C):
    new_list = list()

    if solar_station==ac.ABB_Solarstation.C:
        path = ac.c_img_path

    elif solar_station==ac.ABB_Solarstation.MS:
        path =ac.ms_img_path
    else:
        raise ValueError("Wrong solarstation input in convert_to_tfrecord_paths")

    for day_path in data_list:
        tf_name = os.path.basename(day_path) + '-paths' + "I" + str(FLAGS.image_num_per_sample) + "S" + str(
            FLAGS.strides) + str(
            FLAGS.image_name_suffix) + ".tfrecords"
        full_path = os.path.join(path,os.path.basename(day_path))
        new_list.append(os.path.join(full_path, tf_name))

    return new_list

def custom_abs_max_loss(labels, predictions, weights=1.0,max_loss_weight=0.5):
    with tf.variable_scope("custom_abs_max__loss"):
        residuals = tf.abs(tf.subtract(labels, predictions, name="loss_subtraction"))  # batchsize x 11
        max_residuals = tf.reduce_max(residuals,axis=1)
        residual_mean = tf.reduce_mean(residuals, axis=1)  # mean over 11 predictions  => batchsizex1
        cumulated_loss = (max_loss_weight)*max_residuals+(1-max_loss_weight)*residual_mean
        weighted_mean_max_residuals = tf.multiply(cumulated_loss, weights)
        loss = tf.reduce_mean(weighted_mean_max_residuals)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss

######################################################################################
##NETWORK
######################################################################################



def _configure_network(input_batch, labels, is_training):

    current_irradiance=tf.reshape(labels[:,1],[-1,1])


    inputs, end_points_input = input_layer_factory[FLAGS.input_layer](input_batch,
                                                                      0.0,
                                                                      is_training=is_training,
                                                                      reuse=None)

    first_layer_weights = None  # for layer visualization


    if FLAGS.network_architecture == 'regression_resnetV218':
        print("Network Architecture:",'regression_resnetV218')
        network_output, end_points_network = network_factory['regression_resnetV218'](inputs, 0.0 ,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_18/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())


    elif FLAGS.network_architecture == 'regression_resnetV218_bottleneck':
        print("Network Architecture:",'regression_resnetV218_bottleneck')
        network_output, end_points_network = network_factory['regression_resnetV218_bottleneck'](inputs, 0.0,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_18/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())


    elif FLAGS.network_architecture == 'regression_resnetV218_irr':
        print("Network Architecture:",'regression_resnetV218_irr')
        network_output, end_points_network = network_factory['regression_resnetV218_irr'](inputs,current_irradiance, 0.0,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.irr_pooling_layer,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_18/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_resnetV218_irr_bottleneck':
        print("Network Architecture:",'regression_resnetV218_irr')
        network_output, end_points_network = network_factory['regression_resnetV218_irr_bottleneck'](inputs,current_irradiance, FLAGS.weight_decay,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.irr_pooling_layer,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_18/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_resnetV218_nopool':
        print("Network Architecture:", 'regression_resnetV218_nopool')
        network_output, end_points_network = network_factory['regression_resnetV218_nopool'](inputs, 0.0,
                                                                                             FLAGS.batch_norm_decay,
                                                                                             FLAGS.batch_norm_epsilon,
                                                                                             FLAGS.batch_norm_scale,
                                                                                             FLAGS.global_pool,
                                                                                             FLAGS.output_stride,
                                                                                             FLAGS.spatial_squeeze,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("resnet_v2_18_nopool/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())


    elif FLAGS.network_architecture == 'regression_resnetV250':
        print("Network Architecture:",'regression_resnetV250')
        network_output, end_points_network = network_factory['regression_resnetV250'](inputs,0.0,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_50/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())
    elif FLAGS.network_architecture == 'regression_simple_dqnnet':
        print("Network Architecture:", 'regression_simple_dqnnet')
        network_output, end_points_network = network_factory['regression_simple_dqnnet'](inputs, FLAGS.simple_dqn_outputs,0.0,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_resnetV250_irr':
        print("Network Architecture:",'regression_resnetV250_irr')
        network_output, end_points_network = network_factory['regression_resnetV250_irr'](inputs,current_irradiance, 0.0,
                                                                                      FLAGS.batch_norm_decay,
                                                                                      FLAGS.batch_norm_epsilon,
                                                                                      FLAGS.batch_norm_scale,
                                                                                      FLAGS.global_pool,
                                                                                      FLAGS.output_stride,
                                                                                      FLAGS.spatial_squeeze,
                                                                                      FLAGS.irr_pooling_layer,
                                                                                      FLAGS.include_root_block,
                                                                                      is_training=is_training,
                                                                                      reuse=None)

        with tf.variable_scope("resnet_v2_50/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_irr':
        print("Network Architecture:", 'regression_simple_dqnnet_irr')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_irr'](inputs, current_irradiance, FLAGS.simple_dqn_outputs,0.0,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_do':
        print("Network Architecture:", 'regression_simple_dqnnet_do')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_do'](inputs, FLAGS.simple_dqn_outputs,FLAGS.dqn_keep_prob,0.0,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_bn':
        print("Network Architecture:", 'regression_simple_dqnnet_bn')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_bn'](inputs,
                                                                                            FLAGS.simple_dqn_outputs,
                                                                                            0.0,
                                                                                            FLAGS.dqn_batch_norm_decay,
                                                                                            FLAGS.dqn_batch_norm_epsilon,
                                                                                            FLAGS.dqn_batch_norm_scale,
                                                                                            is_training=is_training,
                                                                                            reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    else:
        raise ValueError('Invalid network architecture')

    print("output_layer",  FLAGS.output_layer)
    if FLAGS.output_layer == 'linear11_irr'  or FLAGS.output_layer == 'linear31_irr':

        predictions, end_points_output = output_layer_factory[FLAGS.output_layer](network_output, current_irradiance,
                                                                                  0.0,
                                                                                  is_training=is_training, reuse=None)
    else:
        predictions, end_points_output = output_layer_factory[FLAGS.output_layer](network_output,
                                                                                  0.0,
                                                                                  is_training=is_training, reuse=None)

    end_points = {**end_points_input, **end_points_output, **end_points_network}

    return predictions, end_points, first_layer_weights



########################################################################################





def _get_train_val_test_sets(day_list, train_size=0.6, validation_size=0.1, test_size=0.3, seed=1):
    size = len(day_list)

    if abs(train_size) + abs(validation_size) + abs(test_size) != 1:
        raise ValueError("Illegal train/test/validation split needs to add up to 1")

    random.seed(seed)
    random.shuffle(day_list)

    if train_size == 0.0:
        print("Training size is 0, returning full Test set")
        test_list = day_list
        return list(), list(), sorted(test_list)

    size_train = int(math.floor(size * train_size))
    train_list = day_list[:size_train]
    residual_list = day_list[size_train:]

    if validation_size == 0.0:
        validation_list = list()
        test_list = residual_list
        return sorted(train_list), validation_list, sorted(test_list)

    if test_size == 0.0:
        test_list = list()
        validation_list = residual_list
        return sorted(train_list), sorted(validation_list), test_list

    size = len(residual_list)
    size_validation = int(math.floor(size * validation_size))

    validation_list = residual_list[:size_validation]
    test_list = residual_list[size_validation:]

    return sorted(train_list), sorted(validation_list), sorted(
        test_list)
    # they are sorted again, because the actual shuffling of days happens in the string_input_producer
    # this allows training/prediction on ordered data if needed (see flags)


def eval_once(saver, summary_writer, loss_op, pred_op, label_op, full_label_op, summary_op, path_op,eval_example_num):
    #device_count = {'GPU': 0},
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=gpu_options
    )
    with tf.Session(config=config) as sess:

        # SETUP
        print("Checkpoint folder:",FLAGS.cpk_dir)

        if FLAGS.cpk_dir:
            cpk_path = FLAGS.cpk_dir
            print("Checkpoint file:", cpk_path)
            saver.restore(sess, cpk_path)
            global_step = cpk_path.split('/')[-1].split('-')[-1]
            print("Training step of checkpoint:", global_step)
        else:
            ValueError("Input path to checkpoint")


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, daemon=True, start=True)
        try:

            num_iter = int(eval_example_num) // int(FLAGS.test_batch_size)

            losses = list()
            step = 1
            data_list = list()
            index_list = list()
            loss_list = list()

            bad_data_list = list()
            bad_index_list = list()


            while step <= num_iter and not coord.should_stop():
                l, la,fla, pre, pa = sess.run([loss_op, label_op, full_label_op,pred_op, path_op])
                losses.append(l)
                print("Eval iteration: " + str(step) + "/"+str(num_iter)+ ", step loss: ", l)


                image_keys = [image_key_creator(image_name.decode("utf-8").rsplit("/",1)[1]) for image_name in pa[:,-1]]
                index_list.extend(image_keys)
                data = np.concatenate((pre,la),axis= 1)
                data_list.extend(data)

                loss_batch = np.repeat(l,len(image_keys))
                loss_list.extend(loss_batch)



                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Step loss', simple_value=l)

                summary_writer.add_summary(summary, step)

                if FLAGS.print_outlier:
                    if FLAGS.print_outlier <= l:
                        print(image_keys)
                        print(data)
                        bad_index_list.extend(image_keys)
                        bad_data_list.extend(data)


                step += 1



            avg_loss = sum(losses) / step
            med_loss = np.median(losses)
            std_loss = np.std(losses)

            percentiles = list()

            for i in range(10,100,10):
                percentile = np.percentile(losses,q=i)
                percentiles.append(percentile)

            time_now = str(dt.datetime.now())
            print("####################################################################################")
            print("EVAL: Time: " + time_now + ", Step: " + str(step) + \
                  ", AVG MSE Loss= {:.2f}".format(avg_loss) + \
                  ", Median MSE Loss= {:.2f}".format(med_loss) + \
                  ", Std. dev MSE Loss= {:.2f}".format(std_loss)+ \
                  ", Percentiles MSE Loss= " + str(percentiles)+\
                  ", Mean change test losses= " + "Not meaningful here")

            with open(os.path.join(FLAGS.eval_dir, "eval_results.csv"), "w+") as f:
                w = csv.writer(f)
                w.writerow([str(dt.datetime.now())])
                for key, val in [("AVG MSE",avg_loss),("MED MSE",med_loss),("STD MSE",std_loss),("PERC MED",str(percentiles))]:
                    w.writerow([key, val])

            if FLAGS.write_predictions_to_csv:
                pd_path = os.path.join(FLAGS.eval_dir, "eval_predictions.csv")
                bad_pd_path = os.path.join(FLAGS.eval_dir, "eval_bad_predictions.csv")
                loss_pd_path = os.path.join(FLAGS.eval_dir, "eval_prediction_losses.csv")

                print("Writing predictions to:", pd_path)

                print(len(data_list),len(index_list))

                data_df = pd.DataFrame(data=np.array(data_list), index=np.array(index_list),
                                       columns=np.array(eval_file_columns)).astype(float).sort_index()

                data_df.to_csv(pd_path, index=True)

                print("Writing loss predictions to:", loss_pd_path)
                loss_data_df = pd.DataFrame(data=np.array(loss_list), index=np.array(index_list),
                                       columns=np.array(['MSE Loss'])).astype(float).sort_index()


                loss_data_df.to_csv(loss_pd_path, index=True)

                if FLAGS.print_outlier:
                    if len(bad_data_list) > 0:
                        print("Writing bad predictions to:", bad_pd_path)
                        bad_data_df = pd.DataFrame(data=np.array(bad_data_list), index=np.array(bad_index_list),
                                           columns=np.array(eval_file_columns)).astype(float).sort_index()

                        bad_data_df.to_csv(bad_pd_path, index=True)
                    else:
                        print("No outliers found")




        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


######################################################################################
##MAIN
######################################################################################



def main(_):
    print(get_available_cpus())
    print(get_available_gpus())

    if tf.gfile.Exists(FLAGS.eval_dir):
        if FLAGS.overwrite_existing_dir:
            tf.gfile.DeleteRecursively(FLAGS.eval_dir)
        else:
            raise ValueError("Evaluation directory exists already")

    os.makedirs(FLAGS.eval_dir)

    with open(os.path.join(FLAGS.eval_dir, "eval_model_info.csv"), "w+") as f:
        w = csv.writer(f)
        w.writerow([str(dt.datetime.now())])
        for key, val in FLAGS.__flags.items():
            w.writerow([key, val])

    tf.logging.set_verbosity(tf.logging.INFO)

    ##############################################################
    # Define ABB Input Pipeline. use QUEUE RUNNERS#
    ##############################################################

    day_list = ABBTFInputPipeline.create_tfrecord_paths(suffix=FLAGS.image_name_suffix,
                                                        img_nr=FLAGS.image_num_per_sample, strides=FLAGS.strides)

    if FLAGS.use_manual_set_selector:
        print("reading train_list.out")
        with open('test_list.out') as f:
            test_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))


        train_list=list()

    else:
        print("creating new test_list.out")
        train_list, _, test_list = _get_train_val_test_sets(day_list, FLAGS.train_set_size, FLAGS.validation_set_size,
                                                            FLAGS.test_set_size,
                                                            FLAGS.train_val_test_split_seed)  # randomly shuffles
    print("copying test_list.out")
    copyfile("test_list.out", os.path.join(FLAGS.eval_dir,'test_list.out'))

    print("Number of days in test set", len(test_list))

    num_test_samples = get_nr_of_samples_in_sets(test_list)
    eval_example_num = FLAGS.eval_example_num
    if FLAGS.eval_example_num is None:
        eval_example_num = num_test_samples


    abb_input = ABBTFInputPipeline(train_list, test_list, resized_image_width=FLAGS.image_width_resize,
                                   resized_image_height=FLAGS.image_height_resize, image_height=FLAGS.image_height,
                                   image_width=FLAGS.image_width,
                                   difference_images=FLAGS.difference_images,
                                   image_channels=FLAGS.image_channels,
                                   train_batch_size=0,
                                   test_batch_size=FLAGS.test_batch_size,
                                   img_num_per_sample=FLAGS.image_num_per_sample,
                                   label_key_list=label_key_list, num_test_queuerunners=FLAGS.num_test_queuerunners,
                                   num_train_queuerunners=0,
                                   capacity_batches_in_train_output_queue=0,
                                   capacity_batches_in_test_output_queue=FLAGS.capacity_batches_in_test_output_queue)

    test_queue, test_shape_img, test_shape_label, test_shape_pl = abb_input.setup_test_queue(
        image_preprocessor=custom_preprocessor,
        num_epochs=None,
        batch_size=FLAGS.test_batch_size,
        stack_axis=FLAGS.stack_axis,
        shuffle_days_in_input_queue=FLAGS.shuffle_days_in_test_input_queue,
        shuffle_batches_in_output_queue=FLAGS.shuffle_batches_in_test_output_queue,
        min_batches_in_shuffle_queue=FLAGS.min_batches_in_test_queue,
        num_of_tfrecord_readers=FLAGS.num_of_test_tfrecord_readers)

    images, labels, paths = test_queue.dequeue()

    tf_global_step = slim.create_global_step()

    ####################
    # Define the network #
    ####################
    predictions, end_points, first_layer_weights = _configure_network(images,labels, tf.constant(False))

    full_labels = labels

    # Irradiance labels each 20 seconds"
    if FLAGS.prediction_nr == 31:
        labels = tf.reshape(full_labels[:, 1:32], (-1, 31))
    elif FLAGS.prediction_nr == 11:
        labels = tf.reshape(full_labels[:, 1:32:3], (-1, 11))
    else:
        raise ValueError("Illegal prediction_nr set, only 11 or 31 possible, set correct outputlayer!")

    print('labels_shape: ', labels.get_shape())

    if FLAGS.loss_function == 'abs':
        loss = tf.losses.absolute_difference(labels, predictions, scope="loss")

    elif FLAGS.loss_function == 'abs_max':
        loss = custom_abs_max_loss(labels, predictions, max_loss_weight=FLAGS.abs_max_weight)

    elif FLAGS.loss_function == 'mse':
        loss = tf.losses.mean_squared_error(labels, predictions, scope="loss")
    else:
        raise ValueError("Illegal loss function")


    #############################
    # Initial TensorBoard summaries #
    #############################

    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    while True:
        eval_once(saver, summary_writer, loss, predictions, labels,full_labels, summary_op, paths,eval_example_num)
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.eval_interval_secs)


#######################################################################################################################
if __name__ == '__main__':
    print("Start EVAL")
    tf.app.run()

import sys

sys.path.append('/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/')
sys.path.append('/home/dladmin/Documents/arthurma/shared/dlabb')

import tensorflow as tf
import math
import datetime as dt
import time
import numpy as np
import csv
import os
import random
import pandas as pd
from pprint import pprint

from tensorflow.contrib import learn

from tensorflow.python.tools import inspect_checkpoint as inspector

slim = tf.contrib.slim
from abb_deeplearning.abb_neuralnet_helpers.nn_sv_input_pipeline_class_low_memory_train_test import ABBTFInputPipeline
from tensorflow.contrib.layers.python.layers import initializers
from  models.slim.deployment import model_deploy
from  models.slim.nets import nets_factory
import skimage.viewer as skw
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
from abb_deeplearning.abb_neuralnet_helpers import vgg_preprocessing
from abb_deeplearning.abb_neuralnet_helpers import custom_preprocessing
from abb_deeplearning.abb_neuralnet_helpers import weight_visualizer
from net_input_layers import input_layer_factory
from net_output_layers import output_layer_factory
from net_network_layers import network_factory
from shutil import copyfile

"""
Adapted ResNet V2  with custom input pipeline and changed output/loss layer. 
"""

FLAGS = tf.app.flags.FLAGS

#######################
# Description#
#######################

tf.app.flags.DEFINE_string(
    'description', 'Resnet still in testing phase',
    'description of experiment will be printed to csv file')

#######################
# Labels trainset statistics #
#######################
value_label = ["VF"]
irr_label = ["IRR"+str(i) for i in range(31)] #IRR0 : current, IRR1 current+20s ,IRR30: 10 minutes
mpc_label = ["MPC"+str(i) for i in range(31)]
sc_label = ["SC"+str(i) for i in range(31)]
ch_label = ["CH"+str(i) for i in range(31)]
b_label = ["B"]
c_label = ["C"]

label_key_list = value_label+irr_label+mpc_label+sc_label+ch_label+b_label+c_label


print(label_key_list)

#training_set_statistics: THESE NEED TO BE RECALCULATED WHEN THE TRAINING SET CHANGES, USE SET_STATISTICS...
tr_sample_nr = 714327
#samples that have n number of changes according to the label C 0,1,2,3,4,5,6,7,8,9
tr_change=[535600, 57488, 59418, 29881, 19364, 7896, 3419, 1014, 226, 21]
tr_change_share = [c/tr_sample_nr for c in tr_change]
tr_weights = [tr_change_share[0]/s for s in tr_change_share] #weights for each nr of change  relative to no change (z.b. 535600 no changes weight = 1.0)
changes = 57488+2*59418+3*29881+4*19364+5*7896+6*3419+7*1014+8*226+9*21
no_changes = 535600*11+57488*10+59418*9+29881*8+19364*7+7896*6+3419*5+1014*4+226*3+21*2
ratio_change = changes/(changes+no_changes)
ratio_nochange = 1 - ratio_change
tr_weight_factor = ratio_nochange/ratio_change



#######################
# Dataset and Input, Directories #
#######################

tf.app.flags.DEFINE_bool(  # Don't change...
    'is_training', True,
    'training or evaluation')

tf.app.flags.DEFINE_float(
    'per_process_gpu_memory_fraction', 0.3,
    'fraction of gpu memory used for this process')


tf.app.flags.DEFINE_string(
    'train_dir', '/home/dladmin/Documents/arthurma/runs/default',
    'Directory where checkpoints, info, and event logs are written to.')

tf.app.flags.DEFINE_boolean(
    'overwrite_existing_dir', False,
    'Overwrite existing directories')

tf.app.flags.DEFINE_bool(
    'balance_training_data',False ,
    'Weights the losses of each sample according to the (B)alance flag in the label files. The B flag defines whether there is a change of sunny to cloudy or vice versa in at least some (2) minutes within the next 10 minutes')


tf.app.flags.DEFINE_string(
    'loss_function','abs_max' ,
    'abs or abs_max')

tf.app.flags.DEFINE_float(
    'abs_max_weight',0.5 ,
    'How much to weight maximum term in abs_max relative to abs error')


tf.app.flags.DEFINE_string(
    'loss_weight_type','B' ,
    'B: weights losses according to simple/hard split. a hard sample has 2 or more changes, C: creates weights according to nr of changes in a sample (times a factor tr_weight_factor set above), D: creates tensor of weights where only the specific prediction is weighted instead of the sample as a whole')


tf.app.flags.DEFINE_bool(
    'use_manual_set_selector', True,
    'Create your own train_list.out,validation_list.out,test_list.out files with the paths to the tfrecord files. Files should be in same directory as code.'
    'They will be copied to the corresponding train folder')

# The following parameters can be used to automatically create data set splits
# deactivated if use_manual_set_selector is true
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

#######################
# Model Flags #
#######################

tf.app.flags.DEFINE_string(
    'network_architecture', 'regression_resnetV218_irr',
    'The name of the architecture to train.')  # regression_resnetV218

tf.app.flags.DEFINE_integer('num_epochs', 5, """Epochs""")

tf.app.flags.DEFINE_integer('max_examples_per_epoch', None,  # 100000/batchsize steps
                            """Number of batches to run. None: infer automatically from tf records""")

tf.app.flags.DEFINE_string(
    'input_layer', 'identity', 'input layer selector')

tf.app.flags.DEFINE_float(
    'input_l2_regularizer',0.00004, 'input layer regularizer')

tf.app.flags.DEFINE_string(
    'output_layer', 'linear11_irr', 'input layer selector') #resnet needs linear11 or linear11_irr

tf.app.flags.DEFINE_float(
    'output_l2_regularizer',0.00004, 'output layer regularizer')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# RESNETV2 #
#######################


tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')  # 0.00004

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



tf.app.flags.DEFINE_integer(
    'image_name_suffix', 256,
    'suffix for tfrecords detection')

tf.app.flags.DEFINE_integer(
    'image_height', 256,
    'Height of input image')

tf.app.flags.DEFINE_integer(
    'image_width', 256,
    'Width of input image')

tf.app.flags.DEFINE_integer(
    'image_channels', 3,
    'Channels of input image 3:rgb, if 1 convert to greyscale')

tf.app.flags.DEFINE_integer(  # resize images to this size
    'image_height_resize', 84,
    'Height of input image')

tf.app.flags.DEFINE_integer(
    'image_width_resize', 84,
    'Width of input image')

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
    'image_num_per_sample', 3,
    'Images per sample')

tf.app.flags.DEFINE_integer(
    'strides', 6,
    'Time distance between each image')

tf.app.flags.DEFINE_integer(  # keep 2 others not tested
    'stack_axis', 2,
    'Image stack axis')

# TRAIN QUEUE

tf.app.flags.DEFINE_bool(
    'shuffle_days_in_train_input_queue', True,
    'shuffles the day tfrecords in the test input queue. If this is False, the training will happen ordered by date of day'
    'but may still be random within days')

tf.app.flags.DEFINE_bool(
    'shuffle_batches_in_train_output_queue', True,
    'shuffles batches  in the test output  queue. Will not randomize batches if False. Note: can randomize among days even if,'
    'shuffle_days_in_test_input_queue is set to False! The days will be read in sequentially but there may be overlap between successive dates, leading'
    'to a batch shuffling among days. Shuffling only within days while keeping day order in tact is not possible yet.')

tf.app.flags.DEFINE_integer(
    'min_batches_in_train_queue', 50000,
    'shuffles the day tfrecords in the queue')

tf.app.flags.DEFINE_integer(
    'num_of_train_tfrecord_readers', 5,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_train_queuerunners', 4,
    'Number of queue runners that fill output train fifo queue')

tf.app.flags.DEFINE_integer(
    'capacity_batches_in_train_output_queue', 30,
    'capacity of the train output queue that contains raw images and label batches')

tf.app.flags.DEFINE_integer('train_batch_size', 5, """Images in batch train""")

# VALIDATION QUEUE
tf.app.flags.DEFINE_integer(
    'validation_every_n_steps', 12000,
    'test trained model on validation data set every n steps')



tf.app.flags.DEFINE_bool(
    'shuffle_days_in_test_input_queue', True,
    'shuffles the day tfrecords in the test input queue. If this is False, the training will happen ordered by date of day'
    'but may still be random within days')

tf.app.flags.DEFINE_bool(
    'shuffle_batches_in_test_output_queue', True,
    'shuffles batches  in the test output  queue. Will not randomize batches if False. Note: can randomize among days even if,'
    'shuffle_days_in_test_input_queue is set to False! The days will be read in sequentially but there may be overlap between successive dates, leading'
    'to a batch shuffling among days. Shuffling only within days while keeping day order in tact is not possible yet.')

tf.app.flags.DEFINE_integer(
    'min_batches_in_test_queue', 50000,
    'shuffles the day tfrecords in the queue')

tf.app.flags.DEFINE_integer(
    'num_of_test_tfrecord_readers', 5,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_test_queuerunners', 2,
    'Number of queue runners that fill output test fifo queue')

tf.app.flags.DEFINE_integer(
    'capacity_batches_in_test_output_queue', 10,
    'capacity of the test output queue that contains raw images and label batches')

tf.app.flags.DEFINE_integer('validation_batch_size', 1, """Images in batch validation""") # keep 1!



######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-2, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags and Early Stopping #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'automatic',
    'Specifies how the learning rate is decayed. One of "fixed", "automatic,exponential",'
    ' or "polynomial", automatic uses patience and tolerance factor to reduce learning rate by decay factor if validation loss plateaus')

tf.app.flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.75, 'Learning rate decay factor.')


# polynomial or exponential:

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 10,
    'Number of epochs after which learning rate decays.')

# automatic and early stopping:

tf.app.flags.DEFINE_float(
    'lr_tolerance', 0.5,
    'Reduce learning rate  if the difference between the val los "patience-1" steps ago and the current value is not larger than tolerance"')

tf.app.flags.DEFINE_integer(
    'lr_patience', None,
    'Number of validation losses where val loss did not improve.')

tf.app.flags.DEFINE_float(
    'early_stopping_tolerance', 1,
    'the difference between the val los "patience-1" steps ago and the current value must be lower by "tolerance" value in order to not trigger early stopping')

tf.app.flags.DEFINE_integer(
    'early_stopping_patience', None,
    'If the validation loss was not reduced in the last k validation steps, stop. Off if None')




#######################
# Logging and Saving #
#######################


tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_steps', 1000,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_checkpoint_steps', 22323,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_nr_checkpoints_saved', 100,
    'Maximum number of newest checkpoints that are saved, older ones are deleted.')

#######################
# Checkpoints #
#######################

# "/home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170616_1"
# /home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170620_nopool_4/-156261
# /home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170619_3/-89292
# /home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170620_nopool_4/-156261
# '/home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170620_nopool_abserror_9/-66969
# '/home/dladmin/Documents/arthurma/runs/resnetV218_regression_train_20170621_nopool_abserror_nobal_11/-44646'
tf.app.flags.DEFINE_string(
    'pretrained_checkpoint_path',
    None,
    'The path to a checkpoint from which to fine-tune. Only if restore_latest_checkpoint is false')

tf.app.flags.DEFINE_boolean(
    'restore_latest_checkpoint', False,
    'restore latest checkpoint in train path')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint will ignore missing variables in checkpoint file (optimistic restore).')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Define which scopes should be trained, all if None, separate scopes with ,'
)


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


def _configure_learning_rate(automatic_learning_rate,num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.train_batch_size *
                      FLAGS.num_epochs_per_decay)


    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')


    elif FLAGS.learning_rate_decay_type == 'automatic':
        return automatic_learning_rate
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_variables_to_train():
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def create_initial_summaries(end_points):
    # Gather initial summaries.
    # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries = set()

    # Add summaries for end_points.

    for end_point in end_points:
        x = end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                        tf.nn.zero_fraction(x)))

    for reg_loss in tf.get_collection((tf.GraphKeys.REGULARIZATION_LOSSES)):
        summaries.add(tf.summary.scalar('reg_losses/%s' % reg_loss.op.name, reg_loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))
    # visualize weights

    return summaries


def _image_summary(summaries, image, scope="", max_outputs=5):
    shape = image.get_shape().as_list()
    print("image_summary", shape)
    c = FLAGS.image_channels
    if FLAGS.stack_axis == 2:
        d = int(shape[3] /c)
        [summaries.add(tf.summary.image(str(scope) + 'input_image_' + str(i), image[:, :, :, i * c:(i + 1) * c],
                                        max_outputs=max_outputs)) for i in range(d)]
    else:
        summaries.add(tf.summary.image(str(scope) + 'input_image', image, max_outputs=max_outputs))


def _weight_image_summary(summaries, weights, scope=""):
    # visualization of first convolutional layer
    # weights is of shape (length,width,depth,filters), z.b. (8,8,6,64) for two images with 3 channels each

    if FLAGS.difference_images:
        split_nr = FLAGS.image_num_per_sample-1
    else:
        split_nr = FLAGS.image_num_per_sample

    split_tensors = tf.split(weights, split_nr, axis=2, name="split")  # list of [(8,8,3,64),(8,8,3,64),...]
    filter_cols = list()
    for split in split_tensors:
        padded_filters = tf.pad(split, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]),
                                mode='CONSTANT')  # filter to 10x10x3x64

        padded_filters_shape = padded_filters.get_shape().as_list()  # 10x10x3x64
        trsp_pf = tf.transpose(padded_filters, perm=[3, 0, 1, 2])  # 64x10x10x3
        filter_col = tf.reshape(trsp_pf, shape=[1, -1, padded_filters_shape[1],
                                                padded_filters_shape[2]])  # 1x64x10x10x3 => 1x640x10x3

        filter_cols.append(filter_col)

    stacked_slices = tf.stack(filter_cols)  # 3x1x640x10x3

    trsp_ss = tf.transpose(stacked_slices, perm=[1, 2, 0, 3, 4])

    trsp_ss_shape = trsp_ss.get_shape().as_list()  # 1x640x3x10x3

    weight_image = tf.reshape(trsp_ss, shape=[1, trsp_ss_shape[1], -1, trsp_ss_shape[4]])  # 1x640x30x3
    summaries.add(tf.summary.image(tensor=weight_image, name="weights"))


def _calculate_total_loss(summaries):
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    print('reg_losses: ', len(regularization_losses))
    loss_list = []
    total_loss = None

    if regularization_losses:
        regularization_loss_acc = tf.add_n(regularization_losses, name="accumulated_reg_losses")
        loss_list.append(regularization_loss_acc)
        summaries.add(tf.summary.scalar(name="accumulated_regularization_loss", tensor=regularization_loss_acc))

    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    if losses:
        losses_acc = tf.add_n(losses, name="accumulated_losses")

        loss_list.append(losses_acc)

    if loss_list:
        total_loss = tf.add_n(loss_list, name="total_loss")

    return total_loss, regularization_losses, losses


def update_gradients(update_ops, optimizer, total_loss, variables_to_train, global_step, summaries):
    gradients = optimizer.compute_gradients(total_loss, variables_to_train)

    for grad, var in gradients:
        if grad is not None:
            summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))

    # and returns a train_tensor and summary_op

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    print("update_ops: ", len(update_ops))

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')
        return train_tensor, gradients


def optimistic_restore(session, save_file):
    # only restores variables that are available in checkpoint file, normally this results in an error
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    print("Saved in checkpoint:", len(saved_shapes))
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if
                        var.name.split(':')[0] in saved_shapes])

    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    print("Restored from checkpoint:", len(restore_vars))
    saver = tf.train.Saver(restore_vars, max_to_keep=FLAGS.max_nr_checkpoints_saved)
    saver.restore(session, save_file)


def get_nr_of_samples_in_sets(training_data_paths, validation_data_paths):
    tr = 0
    for fn in training_data_paths:
        for f in tf.python_io.tf_record_iterator(fn):
            tr += 1

    print("Samples in train per epoch: ", tr)
    print("Steps/Batches  in train epoch: ", tr / FLAGS.train_batch_size)

    vs = 0
    for fn in validation_data_paths:
        for _ in tf.python_io.tf_record_iterator(fn):
            vs += 1

    print("Samples in validation per epoch: ", vs)
    print("Steps/Batches  in validation epoch: ", vs / FLAGS.validation_batch_size)

    return tr, vs


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


def calculate_balanced_dataset_weights(training_data_paths, num_train_samples):
    change_weight = 0

    for path in training_data_paths:
        balance_info_path = path.rsplit('.', 1)[0] + '.balanced'
        df = pd.DataFrame.from_csv(path=balance_info_path, index_col=0)
        change_weight += (df['len'] / num_train_samples) * df[
            'ratio']  # get the ratio of samples where change happens, weighted by ratio of samples in this day / samples over all days

    change_weight = change_weight.ix[0]
    same_weight = 1 - change_weight

    print("This train set has following (change/same) distribution:", change_weight, same_weight)

    if change_weight == 0 or same_weight == 0:
        print("Warning: unbalanced train set (change/same), balancing not working:", change_weight, same_weight)
        same_weight, change_weight = 1, 1
    else:

        if change_weight <= same_weight:  # increase loss to increase gradient size as well. Don't create weights smaller than 1
            change_weight = (same_weight / change_weight)
            same_weight = 1
        else:
            same_weight = same_weight / change_weight
            change_weight = 1

    print("calculated loss weights for balancing (change/same):", str(change_weight) + ":" + str(same_weight))

    return change_weight, same_weight


def _create_weight_tensor_B(labels, change_weight, same_weight, is_training):
    weights = tf.reshape(labels[:, -2], (-1, 1))  # get B label 0 or 1   [batch_size,1]
    weights = tf.cast(weights, dtype=tf.float32)
    change_weight_constants = tf.ones_like(weights) * tf.constant(change_weight,
                                                                  dtype=tf.float32)  # only one vector * weight for changeing samples
    same_weight_constants = tf.ones_like(weights) * tf.constant(same_weight,
                                                                dtype=tf.float32)  # only one vector * weight for non changing samples
    zero_comparison = tf.equal(weights, tf.constant(0.))
    loss_weights_training = tf.where(zero_comparison, same_weight_constants,
                                     change_weight_constants)  # if label is 0 select element from non changing vector, else from changing
    loss_weights_validation = tf.ones_like(weights)

    loss_weights = tf.cond(is_training, lambda: loss_weights_training, lambda: loss_weights_validation)
    return loss_weights  # [batch_size,1]



def _create_weight_tensor_C(labels, is_training):
    change_weight = tr_weight_factor
    weights = tf.reshape(labels[:, -1], (-1, 1))  # get C label 0 or 1   [batch_size,1]
    change_weight_constants = tf.cast(weights, dtype=tf.float32)*change_weight # vector with nr of changes. 0 if no change up to 10 changes, weight by factor of change
    same_weight_constants = tf.ones_like(weights)
    zero_comparison = tf.equal(change_weight_constants, tf.constant(0.))
    loss_weights_training = tf.where(zero_comparison, same_weight_constants,
                                     change_weight_constants)  # if label is 0 select element from non changing vector, else from changing
    loss_weights_validation = tf.ones_like(weights)

    loss_weights = tf.cond(is_training, lambda: loss_weights_training, lambda: loss_weights_validation)
    return loss_weights  # [batch_size,1]


def _create_weight_tensor_D(labels, is_training):
    # label is batch_size x nr_labels
    change_weight = tr_weight_factor
    weights = tf.reshape(labels[:, 34:45], (-1, 11))  # extract change labels, they are either 0 or 1 for all 11 labels
    loss_weights = weights*change_weight # at each position where there is a one, multiply by weight, labels with many changes get higher loss

    same_weight_constants = tf.ones_like(weights)
    zero_comparison = tf.equal(weights, tf.constant(0.))
    loss_weights_training = tf.where(zero_comparison, same_weight_constants,
                                     loss_weights)

    loss_weights_validation = tf.ones_like(weights)
    loss_weights = tf.cond(is_training, lambda: loss_weights_training, lambda: loss_weights_validation)
    return loss_weights  # [batch_size,1]





def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

def custom_abs_loss(labels,predictions,weights):
    with tf.variable_scope("custom_abs_loss"):
        residuals = tf.abs(tf.subtract(labels,predictions,name="loss_subtraction")) #batchsize x 11
        residual_mean = tf.reduce_mean(residuals,axis=1) # mean over 11 predictions  => batchsizex1
        weighted_mean_residuals = tf.multiply(residual_mean,weights)
        loss = tf.reduce_mean(weighted_mean_residuals)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return loss

def custom_abs_max_loss(labels, predictions, weights,max_loss_weight):
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
                                                                      FLAGS.input_l2_regularizer,
                                                                      is_training=is_training,
                                                                      reuse=None)

    first_layer_weights = None  # for layer visualization


    if FLAGS.network_architecture == 'regression_resnetV218':
        print("Network Architecture:",'regression_resnetV218')
        network_output, end_points_network = network_factory['regression_resnetV218'](inputs, FLAGS.weight_decay,
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
        network_output, end_points_network = network_factory['regression_resnetV218_irr'](inputs,current_irradiance, FLAGS.weight_decay,
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
        network_output, end_points_network = network_factory['regression_resnetV218_nopool'](inputs, FLAGS.weight_decay,
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
        network_output, end_points_network = network_factory['regression_resnetV250'](inputs, FLAGS.weight_decay,
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

    elif FLAGS.network_architecture == 'regression_resnetV250_irr':
        print("Network Architecture:",'regression_resnetV250_irr')
        network_output, end_points_network = network_factory['regression_resnetV250_irr'](inputs,current_irradiance, FLAGS.weight_decay,
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


    elif FLAGS.network_architecture == 'regression_simple_dqnnet':
        print("Network Architecture:", 'regression_simple_dqnnet')
        network_output, end_points_network = network_factory['regression_simple_dqnnet'](inputs, FLAGS.simple_dqn_outputs,FLAGS.dqn_weight_decay,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_irr':
        print("Network Architecture:", 'regression_simple_dqnnet_irr')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_irr'](inputs, current_irradiance, FLAGS.simple_dqn_outputs,FLAGS.dqn_weight_decay,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_do':
        print("Network Architecture:", 'regression_simple_dqnnet_do')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_do'](inputs, FLAGS.simple_dqn_outputs,FLAGS.dqn_keep_prob,FLAGS.dqn_weight_decay,
                                                                                             is_training=is_training,
                                                                                             reuse=None)

        with tf.variable_scope("simple_dqnnet/conv1", reuse=True):
            first_layer_weights = tf.get_variable("weights")
            print("First layer weights:", first_layer_weights.get_shape())

    elif FLAGS.network_architecture == 'regression_simple_dqnnet_bn':
        print("Network Architecture:", 'regression_simple_dqnnet_bn')
        network_output, end_points_network = network_factory['regression_simple_dqnnet_bn'](inputs,
                                                                                            FLAGS.simple_dqn_outputs,
                                                                                            FLAGS.dqn_weight_decay,
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
    if FLAGS.output_layer == 'linear11_irr' or FLAGS.output_layer == 'linear31_irr':

        predictions, end_points_output = output_layer_factory[FLAGS.output_layer](network_output, current_irradiance,
                                                                                  FLAGS.output_l2_regularizer,
                                                                                  is_training=is_training, reuse=None)
    else:
        predictions, end_points_output = output_layer_factory[FLAGS.output_layer](network_output,
                                                                                  FLAGS.output_l2_regularizer,
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


######################################################################################
##MAIN
######################################################################################



def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        if FLAGS.overwrite_existing_dir:
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        else:
            raise ValueError("Train directory exists already")

    os.makedirs(FLAGS.train_dir)

    with open(os.path.join(FLAGS.train_dir, "train_model_info.csv"), "w+") as f:
        w = csv.writer(f)
        w.writerow([str(dt.datetime.now())])
        for key, val in FLAGS.__flags.items():
            w.writerow([key, val])

    print("Start main")

    tf.logging.set_verbosity(tf.logging.INFO)

    ##############################################################
    # Define ABB Input Pipeline. use QUEUE RUNNERS#
    ##############################################################

    day_list = ABBTFInputPipeline.create_tfrecord_paths(suffix=FLAGS.image_name_suffix,
                                                        img_nr=FLAGS.image_num_per_sample, strides=FLAGS.strides)

    if FLAGS.use_manual_set_selector:
        print("reading train_list.out")
        with open('train_list.out') as f:
            train_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))

        print("reading validation_list.out")
        with open('validation_list.out') as f:
            validation_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))

        print("reading test_list.out")
        with open('test_list.out') as f:
            test_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))






    else:
        train_list, validation_list, test_list = _get_train_val_test_sets(day_list, FLAGS.train_set_size,
                                                                          FLAGS.validation_set_size,
                                                                          FLAGS.test_set_size,
                                                                          FLAGS.train_val_test_split_seed)


    with open(os.path.join(FLAGS.train_dir, 'train_list.out'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    with open(os.path.join(FLAGS.train_dir, 'validation_list.out'), 'w') as f:
        for item in validation_list:
            f.write("%s\n" % item)
    with open(os.path.join(FLAGS.train_dir, 'test_list.out'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % item)

    print("Number of days", len(day_list))
    print("Number of days in train", len(train_list))
    print("Number of days in validation", len(validation_list))
    print("Number of days in test", len(test_list))
    print("Train batch size", FLAGS.train_batch_size)
    print("Validation batch size", FLAGS.validation_batch_size)

    examples_per_epoch = FLAGS.max_examples_per_epoch

    num_train_samples, num_val_samples = get_nr_of_samples_in_sets(train_list, validation_list)

    if FLAGS.max_examples_per_epoch is None:
        examples_per_epoch = num_train_samples

    change_weight, same_weight = 1, 1  # weights of samples where there are enough changes from cloudy to sunny or vice versa, and weights in case there is no change
    if FLAGS.balance_training_data:  # find the weights by looking at the distribution of data where change happens in the next 10 min or not [B] label
        change_weight, same_weight = calculate_balanced_dataset_weights(train_list, num_train_samples)

    # TODO: ugly hack, in case the validation list is empty, the string_input_producer would crash. This prevents it, given
    # that the train set  is not zero. Note that validation is switched off in case the validationlist is empty! (see session below)
    do_validation = True
    if len(validation_list) == 0:
        print("Validation is switched off!")
        validation_list = train_list  # to prevent exception. Ugly =(
        do_validation = False

    abb_input = ABBTFInputPipeline(train_list, validation_list, resized_image_width=FLAGS.image_width_resize,
                                   resized_image_height=FLAGS.image_height_resize, image_height=FLAGS.image_height,
                                   image_width=FLAGS.image_width,
                                   difference_images=FLAGS.difference_images,
                                   image_channels=FLAGS.image_channels,
                                   train_batch_size=FLAGS.train_batch_size,
                                   test_batch_size=FLAGS.validation_batch_size,
                                   img_num_per_sample=FLAGS.image_num_per_sample,
                                   label_key_list=label_key_list, num_train_queuerunners=FLAGS.num_train_queuerunners,
                                   num_test_queuerunners=FLAGS.num_test_queuerunners,
                                   capacity_batches_in_train_output_queue=FLAGS.capacity_batches_in_train_output_queue,
                                   capacity_batches_in_test_output_queue=FLAGS.capacity_batches_in_test_output_queue)

    train_queue, train_shape_img, train_shape_label, train_shape_pl = abb_input.setup_train_queue(
        image_preprocessor=custom_preprocessor,
        num_epochs=None,
        batch_size=FLAGS.train_batch_size,
        stack_axis=FLAGS.stack_axis,
        shuffle_days_in_input_queue=FLAGS.shuffle_days_in_train_input_queue,
        shuffle_batches_in_output_queue=FLAGS.shuffle_batches_in_train_output_queue,
        min_batches_in_shuffle_queue=FLAGS.min_batches_in_train_queue,
        num_of_tfrecord_readers=FLAGS.num_of_train_tfrecord_readers)

    validation_queue, test_shape_img, test_shape_label, test_shape_pl = abb_input.setup_test_queue(
        image_preprocessor=custom_preprocessor,
        num_epochs=None,
        batch_size=FLAGS.validation_batch_size,
        stack_axis=FLAGS.stack_axis,
        shuffle_days_in_input_queue=FLAGS.shuffle_days_in_test_input_queue,
        shuffle_batches_in_output_queue=FLAGS.shuffle_batches_in_test_output_queue,
        min_batches_in_shuffle_queue=FLAGS.min_batches_in_test_queue,
        num_of_tfrecord_readers=FLAGS.num_of_test_tfrecord_readers)

    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

    q_selector = tf.cond(is_training,
                         lambda: tf.constant(0),
                         lambda: tf.constant(1))

    q = tf.QueueBase.from_list(q_selector, [train_queue, validation_queue])

    images, labels, paths = q.dequeue()

    print("Image shape", images.get_shape())

    ##############################################################
    # END: INPUT PIPELINE#
    ##############################################################



    global_step = slim.create_global_step()

    ####################
    # Define the network #
    ####################
    predictions, end_points, first_layer_weights = _configure_network(images,labels, is_training)

    if FLAGS.loss_weight_type =='C' and FLAGS.balance_training_data:
        loss_weights = _create_weight_tensor_C(labels, is_training)
    elif FLAGS.loss_weight_type =='D' and FLAGS.balance_training_data:
        loss_weights = _create_weight_tensor_D(labels, is_training)
    else:
        loss_weights = _create_weight_tensor_B(labels, change_weight, same_weight, is_training) # default, creates loss weights of 1 if balancing is switched off




    print("loss weights:",loss_weights.get_shape())

    full_labels = labels

    #Irradiance labels each 20 seconds"
    labels = tf.reshape(labels[:, 1:32], (-1, 31))

    #each minute
    #labels = tf.reshape(labels[:, 1:32:3], (-1, 11))

    print('prediction_shape:', predictions.get_shape())
    print('labels_shape: ', labels.get_shape())

    ####################
    # LOSS #
    ####################

    print("label_shape:", labels.get_shape())

    # loss = tf.losses.mean_squared_error(labels, predictions, scope="loss", weights=loss_weights)

    if FLAGS.loss_function == 'abs':
        loss = tf.losses.absolute_difference(labels, predictions, scope="loss", weights=loss_weights)

    elif FLAGS.loss_function == 'abs_max':
        loss = custom_abs_max_loss(labels, predictions, weights=loss_weights,max_loss_weight=FLAGS.abs_max_weight)

    elif FLAGS.loss_function == 'mse':
        loss = tf.losses.mean_squared_error(labels, predictions, scope="loss", weights=loss_weights)
    else:
        raise ValueError("Illegal loss function")
    #loss = custom_abs_loss(labels, predictions, loss_weights)
    #############################
    # Initial TensorBoard summaries #
    #############################
    train_summaries = create_initial_summaries(end_points=end_points)

    _image_summary(train_summaries, images)
    if first_layer_weights is not None:
        _weight_image_summary(train_summaries, first_layer_weights)

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
        moving_average_variables = slim.get_model_variables()
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
    else:
        moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure and Learning rate. #
    #########################################


    automatic_learning_rate = tf.placeholder(tf.float32,shape=None, name="automatic_learning_rate")

    learning_rate = _configure_learning_rate(automatic_learning_rate,examples_per_epoch, global_step)



    # end_learning_rate = tf.constant(FLAGS.end_learning_rate)
    # learning_rate = tf.cond(tf.greater_equal(end_learning_rate,learning_rate), lambda: tf.identity(end_learning_rate), lambda:tf.identity(learning_rate))


    optimizer = _configure_optimizer(learning_rate)
    train_summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Gather update_ops. These contain, for example,
    # the updates for the batch_norm variables.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)



    if FLAGS.moving_average_decay:
        # Update ops executed locally by trainer.
        update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    # Losses, regularization + output loss
    total_loss, regularization_losses, prediction_loss = _calculate_total_loss(train_summaries)

    train_step, gradients = update_gradients(update_ops, optimizer, total_loss, variables_to_train, global_step,
                                             train_summaries)

    # Merge all summaries together.
    total_train_loss_summary = tf.summary.scalar("total_train_loss", total_loss)
    train_loss_summary = tf.summary.scalar("train_loss", loss)
    validation_loss_summary = tf.summary.scalar("validation_loss", loss)
    merged_train = tf.summary.merge(list(train_summaries), name='train_summary_op')
    # merged_test = tf.summary.merge(list(validation_loss_summary),name="test_summary_op")

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    ###########################
    # Kicks off the training. #
    ###########################

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    param_num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])

    if FLAGS.per_process_gpu_memory_fraction:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,gpu_options = gpu_options)
    else:
        config = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("QUEUE RUNNERS:", threads)
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        print("QUEUE RUNNERS:", queue_runners)

        train_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             sess.graph)
        saver = tf.train.Saver(max_to_keep=FLAGS.max_nr_checkpoints_saved)

        if not FLAGS.restore_latest_checkpoint and FLAGS.pretrained_checkpoint_path is not None:

            if FLAGS.ignore_missing_vars:
                print("Restore pretrained model: Optimistic")
                optimistic_restore(sess, FLAGS.pretrained_checkpoint_path)
            else:
                print("Restore pretrained model")
                saver.restore(sess, FLAGS.pretrained_checkpoint_path)

        if FLAGS.restore_latest_checkpoint:
            print("Restore latest Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_checkpoint_path))

        # TRAINING

        val_data = list()
        lr_float = float(FLAGS.learning_rate)


        beginning_time = time.time()
        start_time = time.time()

        step = sess.run(global_step)
        print("First step:", str(step))
        print("Parameter count:", str(param_num))
        # Keep training until reach max iterations

        if FLAGS.num_epochs is None:
            epochs = 1
        else:
            epochs = FLAGS.num_epochs

        for epoch in range(epochs):
            print("Starting epoch " + str(epoch + 1) + "/" + str(epochs))
            while step < ((epoch + 1) * examples_per_epoch) / FLAGS.train_batch_size:  # one epoch

                examples = step * FLAGS.train_batch_size

                # TRAINING

                step, _, ts, ls, p, l, pa, tls, ttls,lw,full_l = sess.run(
                    [global_step, train_step, total_loss, loss, predictions, labels, paths, train_loss_summary,
                     total_train_loss_summary,loss_weights,full_labels],
                    feed_dict={is_training: True,automatic_learning_rate:lr_float})


                if math.isnan(ts) or math.isinf(ts):
                    raise ValueError('Nan or Inf Loss detected at step ' + str(step))

                if step % FLAGS.log_every_n_steps == 0:
                    # Calculate batch loss

                    curr_time = time.time()
                    duration = curr_time - start_time

                    examples_per_sec = FLAGS.log_every_n_steps * FLAGS.train_batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_every_n_steps)

                    time_now = str(dt.datetime.now())
                    print("####################################################################################")
                    print("Time: " + time_now + ", Epoch: " + str(epoch + 1) + ", Step: " + str(step) + \
                          ", Examples: " + str(examples) + ", Minibatch AE= " + \
                          "{:.6f}".format(ts) + ", Examples/sec: " + "{:.2f}".format(
                        examples_per_sec) + ", sec/Batch: " + "{:.2f}".format(
                        sec_per_batch))

                    train_writer.add_summary(tls, step)
                    train_writer.add_summary(ttls, step)

                    # print("label: " + str(list(np.squeeze(l))))
                    # print("pred: " + str(list(np.squeeze(p))))
                    start_time = time.time()

                if step % FLAGS.save_summaries_steps == 0:
                    summary, gr, lr = sess.run([merged_train, gradients, learning_rate], feed_dict={is_training: True,automatic_learning_rate:lr_float})

                    grad_statistics = list()
                    for grads, vars in gr:
                        grad_step = np.linalg.norm(grads * -lr)
                        var_norm = np.linalg.norm(vars)
                        wg_ratio = grad_step / var_norm
                        grad_statistics.append((wg_ratio))

                    mean_wg_ratio = sum(grad_statistics) / len(grad_statistics)
                    median_wg_ratio = np.median(grad_statistics)
                    max_wg_ratio = max(grad_statistics)
                    min_wg_ratio = min(grad_statistics)

                    summary_gwratio = tf.Summary()
                    summary_gwratio.value.add(tag="mean_wg_ratio", simple_value=mean_wg_ratio)
                    summary_gwratio.value.add(tag="median_wg_ratio", simple_value=median_wg_ratio)
                    summary_gwratio.value.add(tag="max_wg_ratio", simple_value=max_wg_ratio)
                    summary_gwratio.value.add(tag="min_wg_ratio", simple_value=min_wg_ratio)
                    summary_gwratio.value.add(tag="learning_rate", simple_value=lr)
                    # summary_writer.add_summary(summary, step)
                    train_writer.add_summary(summary, step)
                    train_writer.add_summary(summary_gwratio, step)
                    # summary_writer.flush()

                if step % FLAGS.save_checkpoint_steps == 0:
                    print("Checkpoint at step " + str(step))
                    saver.save(sess, FLAGS.train_dir, global_step=step)

                if do_validation:
                    if step % FLAGS.validation_every_n_steps == 0:  # VALIADTION
                        val_step = 0
                        val_loss = 0

                        val_loss_easy = 0;
                        val_loss_difficult = 0;
                        val_step_easy = 0;
                        val_change_loss_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        mean_change_val_losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        val_change_step_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        print("VALIDATION#######################################################")
                        curr_val_time = time.time()
                        while val_step < (int(num_val_samples) // int(FLAGS.validation_batch_size)):
                            """
                            ls, p, l, pa, vls = sess.run([loss, predictions, labels, paths, validation_loss_summary],
                                                     feed_dict={is_training: False})
                            """
                            ls, fl = sess.run([loss,full_labels], feed_dict={is_training: False})
                            val_loss += ls
                            val_step += 1

                            if FLAGS.validation_batch_size == 1:
                                if fl[-1][-2] == 1.0: # B label is 1 => changing label
                                    val_loss_difficult +=ls
                                else:
                                    val_step_easy +=1
                                    val_loss_easy += ls

                                vc_change = int(fl[-1][-1]) # get C label

                                if vc_change < 11 and vc_change >= 0:  # should always be the case given the labels above
                                    val_change_loss_list[vc_change] += ls
                                    val_change_step_list[vc_change] += 1

                        # calculate validation loss mean for all changes (0 to 10)

                        for i, (vc_loss, vc_step) in enumerate(zip(val_change_loss_list, val_change_step_list)):
                            if vc_step > 0:
                                mean_vl = vc_loss / vc_step
                                mean_change_val_losses[i] = mean_vl

                        mean_val_loss_difficult, mean_val_loss_easy = 0,0
                        val_step_difficult = val_step-val_step_easy

                        if FLAGS.validation_batch_size == 1:
                            if not (val_step_easy == 0 or val_step_difficult == 0):
                                mean_val_loss_difficult = val_loss_difficult/val_step_difficult
                                mean_val_loss_easy = val_loss_easy/val_step_easy


                        mean_val_loss = val_loss / val_step
                        end_val_time = time.time()
                        duration = end_val_time - curr_val_time

                        print("Duration and Absolute Error (overall/easy/hard):", duration, mean_val_loss, mean_val_loss_easy,mean_val_loss_difficult)

                        val_data.append(mean_val_loss)

                        summary_writer = train_writer
                        summary = tf.Summary()
                        summary.value.add(tag="mean_validation_loss", simple_value=mean_val_loss)
                        summary.value.add(tag="mean_validation_loss_easy", simple_value=mean_val_loss_easy)
                        summary.value.add(tag="mean_validation_loss_hard", simple_value=mean_val_loss_difficult)
                        summary.value.add(tag="mean_validation_0", simple_value=mean_change_val_losses[0])
                        summary.value.add(tag="mean_validation_1", simple_value=mean_change_val_losses[1])
                        summary.value.add(tag="mean_validation_2", simple_value=mean_change_val_losses[2])
                        summary.value.add(tag="mean_validation_3", simple_value=mean_change_val_losses[3])
                        summary.value.add(tag="mean_validation_4", simple_value=mean_change_val_losses[4])
                        summary.value.add(tag="mean_validation_5", simple_value=mean_change_val_losses[5])
                        summary.value.add(tag="mean_validation_6", simple_value=mean_change_val_losses[6])
                        summary.value.add(tag="mean_validation_7", simple_value=mean_change_val_losses[7])
                        summary.value.add(tag="mean_validation_8", simple_value=mean_change_val_losses[8])
                        summary.value.add(tag="mean_validation_9", simple_value=mean_change_val_losses[9])
                        summary.value.add(tag="mean_validation_10", simple_value=mean_change_val_losses[10])
                        summary_writer.add_summary(summary, step)
                        summary_writer.flush()


                        #EARLY STOPPING AND AUTOMATIC LEARNING RATE
                        es_tolerance = FLAGS.early_stopping_tolerance
                        es_patience = FLAGS.early_stopping_patience
                        lr_tolerance = FLAGS.lr_tolerance
                        lr_patience = FLAGS.lr_patience

                        if lr_patience and lr_tolerance:
                            if len(val_data) >= lr_patience:
                                beg_val = val_data[-lr_patience]
                                end_val = val_data[-1]
                                if beg_val - end_val < lr_tolerance:
                                    print("Reduce automatic learning rate to:",lr_float)
                                    lr_float = lr_float*FLAGS.learning_rate_decay_factor
                                    if lr_float <= FLAGS.end_learning_rate:
                                        lr_float = FLAGS.end_learning_rate

                        if es_patience and es_tolerance:
                            if len(val_data) >= es_patience:
                                beg_val = val_data[-es_patience]
                                end_val = val_data[-1]
                                if beg_val - end_val < es_tolerance:
                                    sys.exit("Early Stopping")





            print("Epoch finished with step " + str(step))

        print("Finished epochs:  " + str(epochs) + " with " + str(step * FLAGS.train_batch_size) + " examples in total")

        end_time = time.time()
        runtime = end_time - beginning_time
        print("Total runtime: " + str(runtime))

        coord.request_stop()
        coord.join(threads)


#######################################################################################################################
if __name__ == '__main__':
    print("Start")
    tf.app.run()

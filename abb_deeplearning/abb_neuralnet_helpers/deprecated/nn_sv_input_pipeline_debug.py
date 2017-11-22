"""
Defines functions for the input pipeline for supervised learning problems. It reads TFRecords
"""
import tensorflow as tf
import numpy as np
import skimage.io as skio
import skimage.viewer as skw
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_r
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import os

###################################################################################
# 1566
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_CHANNELS = 3

default_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)


def decode_single_tfrecord(filename_queue, feature_dict, name, options=default_options):
    # Deprecated, use decode_multiple_tfrecord with nr_of_readers = 1 for same effect
    tfreader = tf.TFRecordReader(options=options)
    key, serialized_value = tfreader.read(filename_queue, name=name)
    features = tf.parse_single_example(serialized_value, features=feature_dict)
    return key, features


def decode_multiple_tfrecord(filename_queue, nr_of_readers, feature_dict, name,
                             options=default_options):  # must use tf.train.shuffle_batch_join afterwards!
    """
    Reads lines from tfrecord files
    :param filename_queue: tensorflow filename queue
    :param nr_of_readers: nr of threads that read from the queue. This allows for randomization across different files (days)
    the more threads, the more different days are randomized. The Filename queue can be shuffled as well, allowing for random day access per epoch
    :param feature_dict: dictionary of label name and type as was used in the definition of the tfrecords file
    :param name: optional name for operation
    :param options: tfrecord reader options. In case data was compressed
    :return: returns a list of file_paths and decoded tf records that can be used to get exampels: (image,label) tensors with get_multiple_examples
    """


    tfreaders = [tf.TFRecordReader(options=options) for _ in range(nr_of_readers)]
    output_list = [tfreader.read(filename_queue) for tfreader in tfreaders]  # creates list [key,serialized_values]
    keys = [output[0] for output in output_list]
    features_list = [tf.parse_single_example(output[1], features=feature_dict) for output in
                     output_list]  # take serialized values from each element
    return keys, features_list


def create_shuffled_data_batch(data, label, batch_size, capacity, num_threads, min_after_dequeue):
    """
    Shuffling of batches. Not to be used with decode_multiple_tfrecords and get_multiple_examples
    """
    data, label = tf.train.shuffle_batch([data, label], batch_size=batch_size, capacity=capacity,
                                         num_threads=num_threads,
                                         min_after_dequeue=min_after_dequeue)

    return data, label


def create_joined_shuffled_data_batch(tensor_list, batch_size, capacity, min_after_dequeue):
    """
     Use with decode_multiple_tfrecords and get_multiple_examples. Takes the tensor_list from get_multiple_examples
    amd returns a batch of images with the labels of the last image, in case there are multiple images per example
    :param tensor_list: list returned by get_multiple_example, contains list of lists. Each inner list hast an image and label tensor
    :param batch_size: nr of examples per batch
    :param capacity: size of batch queue
    :param min_after_dequeue: minimum examples in the batch queue. Important for random shuffling of batches. Better shuffling if larger, but needs more memory
    :return: return a batch with examples
    """
    data, label,paths = tf.train.shuffle_batch_join(tensor_list, batch_size=batch_size, capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)

    return data, label,paths


def create_data_batch(data, label, batch_size, capacity, num_threads):
    """
    Same as create_shuffled_data_batch, but does not shuffle the batches
    """
    data, label = tf.train.batch([data, label], batch_size=batch_size, capacity=capacity,
                                 num_threads=num_threads)


    return data, label


def __prepr_functions_default(image):
    """
    Default function for image processor, just reshapes image to correct  dimensions
    :param image: image tensor
    :return: reshaped image tensor
    """
    # needs to start with reshaping to original size!
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    return image


def __preprocess_jpeg(decoded_image, preprocessor):
    return preprocessor(decoded_image)


def stack_images(image_tensor, axis=0):
    """
    Stacks images according to axis argument
    :param image_tensor: input 4D tensor (nr images, height,width,channel)
    :param axis: 0: stack along height, 1: stack along width, 2: stack along color channel
    :return: reshaped 3D image tensor
    """
    print(image_tensor.get_shape())
    shape = image_tensor.get_shape().as_list()
    height = shape[1]
    width = shape[2]
    channels = shape[3]




    # expects 4D Tensor, (image,height,width,channel)
    if axis == 0:  # stack vertically along rowns
        # t_transp = tf.transpose(image_tensor,perm=[3,1,0,2]) # depth first, reshape will start from the most right dimension (go along height of image)
        return tf.reshape(image_tensor, [-1, width, channels])

    elif axis == 1:  # stack horizontally along columns
        t_transp = tf.transpose(image_tensor, perm=[1, 0, 2,
                                                    3])  # depth first, reshape will start from the most right dimension (go along width of image)
        return tf.reshape(t_transp, [height, -1, channels])

    elif axis == 2:  # stack along color channels
        t_transp = tf.transpose(image_tensor,
                                perm=[1, 2, 0, 3])  # depth first, reshape will start from the most right dimension


        return tf.reshape(t_transp, [height, width, -1])


def get_example(decoded_features, feature_list, img_path_keys=None, image_preprocessor=__prepr_functions_default):
    """
    Use get_multiple_example
    """
    if img_path_keys is None:
        raise ValueError('No image path key list!')

    labels = list()
    for feature in feature_list:
        # print(feature,decoded_features[feature])
        labels.append(decoded_features[feature])

    label_tensor = tf.stack(labels)

    image_paths = [decoded_features[key] for key in img_path_keys]

    with tf.name_scope('decode_image', [image_paths], None):
        image_files = [tf.read_file(path) for path in image_paths]
        dec_image_files = [tf.cast(tf.image.decode_jpeg(file, fancy_upscaling=True, channels=IMAGE_CHANNELS),dtype=tf.float32) for file in
                           image_files]

        preprocessed_images = [__preprocess_jpeg(file, preprocessor=image_preprocessor) for file in dec_image_files]

    image_tensor = tf.stack(preprocessed_images)

    return image_tensor, label_tensor


def get_multiple_example(decoded_features_list, feature_list, img_path_keys=None,
                         image_preprocessor=__prepr_functions_default, stack_axis=0):
    """
    Generates an example, a 3D Tensor that includes all images of a sample. Also saves paths to images in sample for debugging
    :param decoded_features_list: decoded from tfrecords file
    :param feature_list: feature that are encoded in the tfrecords file. Dictionary from string name to type
    :param img_path_keys: names of the image paths that were encoded in the tf records file, dictionary from name to type
    :param image_preprocessor: preprocessing function, applied to all images individually
    :param stack_axis: stacks multiple images in a sample, reduces 4D tensor to 3D tensor (height (0),width (1),channel (2), integer defines along which axis images are stacked
    :return: saves all the tensors that were decoded bei the tfreaders in a tensor list and the paths in a path list. The tensor list is used ti create a shuffled batch
    """
    tensor_list = list()

    for decoded_features in decoded_features_list:
        if img_path_keys is None:
            raise ValueError('No image path key list!')

        labels = list()
        for feature in feature_list:
            # print(feature,decoded_features[feature])
            labels.append(decoded_features[feature])

        label_tensor = tf.stack(labels)

        image_paths = [decoded_features[key] for key in img_path_keys]


        with tf.name_scope('decode_image', [image_paths], None):
            image_files = [tf.read_file(path) for path in image_paths]
            dec_image_files = [tf.cast(tf.image.decode_jpeg(file, fancy_upscaling=True, channels=IMAGE_CHANNELS),dtype=tf.float32) for file in
                               image_files]

            preprocessed_images = [__preprocess_jpeg(file, preprocessor=image_preprocessor) for file in dec_image_files]

        image_tensor = tf.stack(preprocessed_images)

        image_tensor = stack_images(image_tensor, axis=stack_axis)
        #print(image_tensor.get_shape())

        tensor_list.append([image_tensor, label_tensor,image_paths])

    return tensor_list


def create_tfrecord_paths(solar_station=abb_c.ABB_Solarstation.C, date_ranges=None, suffix=384, img_nr=2, strides=1):
    """
    Convenience function to create the paths to the record files, according to the provided date range. This is given as an argument to the tf string producer
    :param solar_station: which solar power station C or MS
    :param date_ranges: date ranges that should be considered, each day has a tf records file, only return files of days within range
    :param suffix: suffix that was given to identify a tfrecords file (given while producing them with nn_tfrecords_creator
    :param img_nr: nr of images in a sample
    :param strides: stride that was set when creating tfrecords
    :return: Returns a list of path strings to the tf record files
    """
    # only datesfor time change tf records
    # use this function to create test and train set
    day_list = abb_r.read_cld_img_day_range_paths(solar_station=solar_station, img_d_tup_l=date_ranges,
                                                  randomize_days=False)
    tf_record_paths = list()
    for day_path, _ in day_list:
        tf_name = os.path.basename(day_path) + '-paths' + "I" + str(img_nr) + "S" + str(strides) + str(
            suffix) + ".tfrecords"
        tf_path = os.path.join(day_path, tf_name)
        tf_record_paths.append(tf_path)

    return tf_record_paths


#####################################################################
img_nr_per_sample = 2
num_epochs = 1
stack_axis = 1
min_batches_in_queue = 25000
batch_size = 10
number_of_tfrecord_readers = 3  # smaller than nr of tf record files in queue!
shuffle_days_in_queue = True
tf_record_file_paths = create_tfrecord_paths(img_nr=2,strides=1)
feature_list = ['VF', 'IRR0', 'IRR1', 'IRR2', 'IRR3', 'IRR4', 'IRR5', 'IRR6', 'IRR7', 'IRR8', 'IRR9', 'IRR10', 'MPC0',
                'MPC1', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10']


def custom_preprocessor(image):
    # needs to start with reshaping to original size!
    # Only nearest_neighbor worked as resize method
    image = tf.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    image = tf.image.resize_images(image,[128,128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.image.adjust_brightness(image,0.1)

    #print(image.get_shape())
    #image = tf.image.per_image_standardization(image)

    # float_image.set_shape([height, width, 3])
    # print(image.get_shape())
    #image = tf.image.central_crop(image,0.8)
    #image = tf.random_crop(image, [120,120, 3])
    #print(image.get_shape())
    # image = tf.image.random_flip_left_right(image)
    # print(image.get_shape())
    return image


#########################################################################################################################

features = {}
for f in feature_list:
    features[f] = tf.FixedLenFeature([], tf.float32)

img_path_keys = list()

for i in range(img_nr_per_sample):
    img_path_keys.append('image_path' + str(i))
    features['image_path' + str(i)] = tf.FixedLenFeature([], tf.string)

# Use multiple TFRecord readers to improve shuffling among daily TFRecord files, prevent having more readers than input files, Exception but does not crash, but may interfere with epochs!
# Use this, not the code above. Set reader_nr to 1 which  is the same as above.
# TODO: TEST Path_list, last check on label to image coherence
# TODO: Preprocessing on consecutive images should be identical. This is not yet the case for random changes to the images, as preprocessing is performed on each image individually
# TODO: get object with all info as interface
# TODO: Proceed as in cifar
with tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(tf_record_file_paths, num_epochs=num_epochs,
                                                shuffle=shuffle_days_in_queue)
    tf_keys, decoded_fs = decode_multiple_tfrecord(filename_queue, number_of_tfrecord_readers, features,
                                               name="tf_records_reader")
    tensor_list = get_multiple_example(decoded_features_list=decoded_fs, feature_list=feature_list,
                                              img_path_keys=img_path_keys, image_preprocessor=custom_preprocessor,
                                              stack_axis=stack_axis)
    imageBatch, labelBatch,path_list = create_joined_shuffled_data_batch(tensor_list, batch_size=batch_size,
                                                           capacity=min_batches_in_queue + 1000,
                                                           min_after_dequeue=min_batches_in_queue)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    for i in range(100):

        p, f, l = sess.run([path_list, imageBatch, labelBatch])
        print(p)
        print(l.shape, l)
        print(f.shape)
        batch_stack = tf.stack(f)
        print(len(batch_stack.eval()))
        for b_image in batch_stack.eval():
            if stack_axis == 2:
                d = int(b_image.shape[2] / 3)  # assume 3 channels
                img_l = [b_image[:, :, i * 3:(i + 1) * 3] for i in range(d)]
                cv = skw.CollectionViewer(img_l)
            else:
                cv = skw.CollectionViewer([b_image])

            cv.show()

    coord.request_stop()
    coord.join(threads)

#######################################################################################################################################################################

# Use only a single TFRecord reader
"""
filename_queue = tf.train.string_input_producer(['/media/data/Daten/img_C/C-2015-08-15/C-2015-08-15-pathsI2S1384.tfrecords'],num_epochs= num_epochs)
tf_key,decoded_f = decode_single_tfrecord(filename_queue,features, name="tf_records_reader")
image,label = get_example(decoded_f,feature_list,img_path_keys=img_path_keys)
#image = stack_images(image,axis=0)
imageBatch,labelBatch = create_shuffled_data_batch(image,label,batch_size=32,capacity=4600,min_after_dequeue=4500,num_threads=2)
"""

"""
image_path,label = create_imgpath_label_tensors(decoded_f,feature_list)
img_path,img_file,label= create_sample_tensors(label,image_path)
img_file,label= create_shuffled_data_batch(img_file,label,batch_size=1,capacity=30,num_threads=2,min_after_dequeue=15)
"""

"""
        batch_stack = tf.stack(f)
        print(len(batch_stack.eval()))
        for b_image in batch_stack.eval():
         #   img.eval()
            img_list = tf.unstack(b_image)
            imgs = [img.eval()for img in img_list]
            cv= skw.CollectionViewer(imgs)

            cv.show()
        """

"""
def decode_jpeg(img_path, label,preprocessor = __prepr_functions_default):
    #image_reader = tf.WholeFileReader()
    #filename_queue = tf.train.string_input_producer([img_path], shuffle=False)
    #img_key, image_file = image_reader.read(filename_queue)
    image_file = tf.read_file(img_path)
    image_file = tf.image.decode_jpeg(image_file,fancy_upscaling=True)
    image_file = tf.reshape(image_file,(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS))
    image_file = __preprocess_jpeg(image_file,preprocessor)
    return img_path,image_file,label
def get_image(decoded_features,img_path_key='image_path'):
    return decoded_features[img_path_key]
def decode_labels(decoded_features,feature_list):
    labels = list()
    for feature in feature_list:
        #print(feature,decoded_features[feature])
        labels.append(decoded_features[feature])


    label_tensor = tf.stack(labels)

    return tf.reshape(label_tensor,(-1,1))
def create_imgpath_label_tensors(decoded_features,feature_list,image_preprocessor=__prepr_functions_default):

    image_path_tensor = get_image(decoded_features)

    label_tensor  = decode_labels(decoded_features,feature_list)

    return  image_path_tensor,label_tensor

def create_sample_tensors(label,img_path,image_preprocessor=__prepr_functions_default):
    img_path, img_file,label = decode_jpeg(img_path,label,preprocessor=image_preprocessor)
    return img_path,img_file,label

"""

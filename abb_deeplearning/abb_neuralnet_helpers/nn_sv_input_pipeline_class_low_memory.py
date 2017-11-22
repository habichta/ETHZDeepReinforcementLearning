"""
Defines functions for the input pipeline for supervised learning problems. It reads TFRecords
Low memory version does not output the decoded images as samples but their paths, decoding and preprocessing after fetching elements from 
Queue. Maybe Build second fifo queue to do this
"""
import tensorflow as tf
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as abb_r
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as abb_c
import threading
import os
import random


###################################################################################
class ABBTFInputPipeline:
    """
    Describes an input pipeline for a specific set of tfrecords. The information that describes these records are fixed 
    class attributes. Other (dynamic) variables for the training and
    evaluation process can be set in the get_preprocessed_input_shuffled_batch method.
    It uses QueueRunners and asynchronous methods to improve the performance and decrease the probability that I/O will 
    be the bottleneck while training/testing
    """
    default_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    def __init__(self, tfrecord_file_path_list, image_height, image_width, image_channels, img_num_per_sample,
                 label_key_list,capacity_batches_in_output_queue=3000,num_queuerunners = 4,tfreader_options=default_options):
        # Fixed arguments that describe a tfrecords file
        self.tfrecord_file_path_list = tfrecord_file_path_list
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = image_height, image_width, image_channels
        self.img_num_per_sample = img_num_per_sample
        self.label_key_list = label_key_list
        self.labels = {}
        self.tfreader_options = tfreader_options


        self.output_queue = tf.FIFOQueue(capacity_batches_in_output_queue,dtypes=[tf.float32,tf.float32,tf.string]) #images,labels,paths
        self.num_queuerunners = num_queuerunners

        for l in self.label_key_list:
            self.labels[l] = tf.FixedLenFeature([], tf.float32)

        self.img_path_keys = list()

        for i in range(self.img_num_per_sample):
            self.img_path_keys.append('image_path' + str(i))
            self.labels['image_path' + str(i)] = tf.FixedLenFeature([], tf.string)

    @staticmethod
    def create_tfrecord_paths(solar_station=abb_c.ABB_Solarstation.C, date_ranges=None, suffix=384, img_nr=2,
                              strides=1):
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

    def decode_single_tfrecord(self, filename_queue, feature_dict, name, options=default_options):
        # Deprecated, use decode_multiple_tfrecord with nr_of_readers = 1 for same effect
        tfreader = tf.TFRecordReader(options=options)
        key, serialized_value = tfreader.read(filename_queue, name=name)
        features = tf.parse_single_example(serialized_value, features=feature_dict)
        return key, features

    def decode_multiple_tfrecord(self, filename_queue, nr_of_readers, feature_dict, name,
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

    def create_shuffled_data_batch(self, data, label, batch_size, capacity, num_threads, min_after_dequeue):
        """
        Shuffling of batches. Not to be used with decode_multiple_tfrecords and get_multiple_examples
        """
        #TODO: Old, see joined_shuffled_data_batch
        data, label = tf.train.shuffle_batch([data, label], batch_size=batch_size, capacity=capacity,
                                             num_threads=num_threads,
                                             min_after_dequeue=min_after_dequeue)

        return data, label

    def create_joined_shuffled_data_batch(self, tensor_list, batch_size, capacity, min_after_dequeue):
        """
         Use with decode_multiple_tfrecords and get_multiple_examples. Takes the tensor_list from get_multiple_examples
        amd returns a batch of images with the labels of the last image, in case there are multiple images per example
        :param tensor_list: list returned by get_multiple_example, contains list of lists. Each inner list hast an image and label tensor
        :param batch_size: nr of examples per batch
        :param capacity: size of batch queue
        :param min_after_dequeue: minimum examples in the batch queue. Important for random shuffling of batches. Better shuffling if larger, but needs more memory
        :return: return a batch with examples
        """
        label, paths = tf.train.shuffle_batch_join(tensor_list, batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)

        return label,paths

    def create_data_batch(self, data, label, batch_size, capacity, num_threads):
        """
        Same as create_shuffled_data_batch, but does not shuffle the batches
        """
        # TODO: Old, see joined_shuffled_data_batch
        data, label = tf.train.batch([data, label], batch_size=batch_size, capacity=capacity,
                                     num_threads=num_threads)

        return data, label

    def __prepr_functions_default(self, image):
        """
        Default function for image processor, just reshapes image to correct  dimensions
        :param image: image tensor
        :return: reshaped image tensor
        """
        # needs to start with reshaping to original size!
        image = tf.image.random_flip_left_right(image)
        image = tf.reshape(image, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS))

        return image



    def stack_images(self, image_tensor, axis=0):
        """
        Stacks images according to axis argument
        :param image_tensor: input 4D tensor (nr images, height,width,channel)
        :param axis: 0: stack along height, 1: stack along width, 2: stack along color channel
        :return: reshaped 3D image tensor
        """
        shape = image_tensor.get_shape().as_list()
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # expects 4D Tensor, (image,height,width,channel)
        if axis == 0:  # stack vertically along rowns
            return tf.reshape(image_tensor, [-1, width, channels])

        elif axis == 1:  # stack horizontally along columns
            t_transp = tf.transpose(image_tensor, perm=[1, 0, 2,
                                                        3])  # depth first, reshape will start from the most right dimension (go along width of image)
            return tf.reshape(t_transp, [height, -1, channels])

        elif axis == 2:  # stack along color channels
            t_transp = tf.transpose(image_tensor,
                                    perm=[1, 2, 0, 3])  # depth first, reshape will start from the most right dimension
            return tf.reshape(t_transp, [height, width, -1])


    def get_multiple_example_lm(self, decoded_features_list, feature_list, img_path_keys=None,
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
        tensor_low_memory_list = list()

        for decoded_features in decoded_features_list:
            if img_path_keys is None:
                raise ValueError('No image path key list!')

            labels = list()
            for feature in feature_list:
                # print(feature,decoded_features[feature])
                labels.append(decoded_features[feature])

            label_tensor = tf.stack(labels)

            image_paths = [decoded_features[key] for key in img_path_keys]




            tensor_low_memory_list.append([label_tensor, image_paths])
        return  tensor_low_memory_list



    def get_preprocessed_input_shuffled_batch(self, image_preprocessor=None, num_epochs=None,
                                              batch_size=32, shuffle_days_in_input_queue=True,
                                              min_batches_in_shuffle_queue=21000, num_of_tfrecord_readers=3,stack_axis=2):

        """
        Interface to the training/testing algorithm. Returns a batch of images
        :param image_preprocessor: function that describes how each image will be transformed individually
        :param num_epochs: epoch numbers in training process. Decides how often a a tfrecords file (for a day) will be put in the input queue
        :param batch_size: nr of sampled (as described in tfrecords file) per batch
        :param shuffle_days_in_input_queue: The daily tfrecord file are put into the input queue in order
        set this to True if shuffling of days is needed (also needs multiple tfrecord_readers, otherwise
        batches will only contain images of a single day
        :param min_batches_in_queue: How many batches are saved in the queue. The larger, the better the randomization
        of samples within a single day. But needs more memory, as decoded images a put onto the queue
        :param num_of_tfrecord_readers: number of concurrent readers that read from the inout queue. Improves randomization
        put can also increase threading overhead. do not use more readers than there are tfrecord files
        :param stack_axis: 0: stack multiple images along height, 1:along width, 2: long color channels
        :return: Returns a batch of samples according to specification. Can be both in order of input files or totally shuffled
        among days and within days.
        """

        for f in self.tfrecord_file_path_list:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        if image_preprocessor is None:
            image_preprocessor = self.__prepr_functions_default
        else:
            image_preprocessor = image_preprocessor

        filename_queue = tf.train.string_input_producer(self.tfrecord_file_path_list, num_epochs=num_epochs,
                                                        shuffle=shuffle_days_in_input_queue)

        tf_keys, decoded_fs = self.decode_multiple_tfrecord(filename_queue, num_of_tfrecord_readers, self.labels,
                                                            name="tf_records_reader", options=self.tfreader_options)
        tensor_low_memory_list = self.get_multiple_example_lm(decoded_features_list=decoded_fs,
                                                           feature_list=self.label_key_list,
                                                           img_path_keys=self.img_path_keys,
                                                           image_preprocessor=image_preprocessor,
                                                           stack_axis=stack_axis)
        label_batch, path_list = self.create_joined_shuffled_data_batch(tensor_low_memory_list, batch_size=batch_size,
                                                                        capacity=min_batches_in_shuffle_queue*2,
                                                                        min_after_dequeue=min_batches_in_shuffle_queue)




        path_batch_list = tf.unstack(path_list)  #list([img1,img2],[img1,img2],...)

        preprocessed_batch = list()
        with tf.name_scope('decode_image', [path_batch_list], None):

            for image_paths in path_batch_list:

               image_files = [tf.read_file(path) for path in tf.unstack(image_paths)] #each image path in a [img1,img2,..]
               dec_image_files = [
                   tf.cast(tf.image.decode_jpeg(file, fancy_upscaling=True, channels=self.IMAGE_CHANNELS), dtype=tf.float32)
                   for file in
                   image_files]

               #seed = tf.random_uniform([1], minval=0, maxval=4000000, dtype=tf.int32, seed=None, name=None)

               preprocessed_images = [image_preprocessor(file) for file in
                                      dec_image_files] #[prepr_img1,prepr_img2,..]
               image_tensor = tf.stack(preprocessed_images) #[prepr_img1,prepr_img2,..] list to tensor

               image_tensor = self.stack_images(image_tensor, axis=stack_axis) #reshape tensor of images along stack axis

               preprocessed_batch.append(image_tensor) #add images sample to batch list

            preprocessed_batch_tensor = tf.stack(preprocessed_batch) #creates tensor that represents a batch of sampled (batch_size, imageshapes)



        shape_img = preprocessed_batch_tensor.shape
        shape_label = label_batch.shape
        shape_pl = path_list.shape




        enqueue_op = self.output_queue.enqueue([preprocessed_batch_tensor,label_batch,path_list])

        qr = tf.train.QueueRunner(self.output_queue,[enqueue_op]* self.num_queuerunners)
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,qr)

        img_batch_tensor,label_batch_tensor,path_list_tensor = self.output_queue.dequeue()

        img_batch_tensor.set_shape(shape_img)
        label_batch_tensor.set_shape(shape_label)
        path_list_tensor.set_shape(shape_pl)



        return img_batch_tensor,label_batch_tensor, path_list_tensor

import tensorflow as tf
import os
import collections

strides=6
image_name_suffix=256
image_num_per_sample=2



def _convert_to_tfrecord_paths(data_list):
    new_list = list()

    for day_path in data_list:
        tf_name = os.path.basename(day_path) + '-paths' + "I" + str(image_num_per_sample) + "S" + str(
            strides) + str(
            image_name_suffix) + ".tfrecords"
        new_list.append(os.path.join(day_path, tf_name))

    return new_list


print("reading train_list.out")
with open('train_list_hard2.out') as f:
    train_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))

print("reading validation_list.out")
with open('validation_list.out') as f:
    validation_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))

print("reading test_list.out")
with open('test_list.out') as f:
    test_list = sorted(_convert_to_tfrecord_paths(f.read().splitlines()))


labels_dict={}

label_key_list = ['VF', 'IRR0', 'IRR1', 'IRR2', 'IRR3', 'IRR4', 'IRR5', 'IRR6', 'IRR7', 'IRR8', 'IRR9', 'IRR10', 'MPC0',
                  'MPC1', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'MPC8', 'MPC9', 'MPC10', 'SC0', 'SC1', 'SC2',
                  'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10', 'CH0', 'CH1', 'CH2',
                  'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9', 'CH10', 'B','C']

for l in label_key_list:
    labels_dict[l] = tf.FixedLenFeature([], tf.float32)

for i in range(image_num_per_sample):
    labels_dict['image_path' + str(i)] = tf.FixedLenFeature([], tf.string)

print(labels_dict)

filename_queue = tf.train.string_input_producer(train_list, num_epochs=1,
                                                            shuffle=False)

tfreader = tf.TFRecordReader()
key,output= tfreader.read(filename_queue)

features_list = tf.parse_single_example(output, features=labels_dict)

labels = list()

for feature in labels_dict.keys():
    # print(feature,decoded_features[feature])
    labels.append(features_list[feature])

#label_tensor = tf.stack(labels)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(log_device_placement=False,gpu_options = gpu_options)

with tf.Session(config=config) as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    c = collections.Counter()
    change_list = list()
    i = 0
    try:



        while i<714327:
         l = sess.run(labels)
         changes = int(l[-3])


         if changes > 7:
             print(l)

         change_list.append(changes)

         i+=1
         if i%10000==0:
            print(i)


    finally:
        print(i)
        c.update(change_list)
        print(dict(c))
        coord.request_stop()

    coord.join(threads)



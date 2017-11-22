import tensorflow as tf

slim = tf.contrib.slim


def simple_dqnnet(inputs, output_size=11, scope=None, is_training=True, reuse=None, regularizer=0.0):
    with tf.variable_scope(scope, 'net', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection,
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(inputs, 32, 8, [4, 4], scope='conv1', padding='SAME')
            net = slim.conv2d(net, 64, 4, [2, 2], scope='conv2', padding='SAME')
            conv_out = slim.flatten(slim.conv2d(net, 64, 3, [1, 1], scope='conv3', padding='SAME'), scope='flatten')
            net = slim.fully_connected(conv_out, 512, scope='fc1')
            prediction = slim.fully_connected(net, output_size, activation_fn=None, scope='fc2')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return prediction, end_points


def simple_dqnnet_do(inputs, output_size=11, keep_prob=0.5, scope=None, is_training=True, reuse=None, regularizer=0.0):
    with tf.variable_scope(scope, 'net', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'



        do_keep_prob = tf.cond(is_training,lambda: tf.constant(keep_prob), lambda: tf.constant(1.0))


        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection,
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(inputs, 32, 8, [4, 4], scope='conv1', padding='SAME')
            net = slim.conv2d(net, 64, 4, [2, 2], scope='conv2', padding='SAME')
            conv_out = slim.flatten(slim.conv2d(net, 64, 3, [1, 1], scope='conv3', padding='SAME'), scope='flatten')
            net = slim.fully_connected(conv_out, 512, scope='fc1')
            net = slim.dropout(net,do_keep_prob, scope='dropout1')
            prediction = slim.fully_connected(net, output_size, activation_fn=None, scope='fc2')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return prediction, end_points


batch_norm_decay = 0.997,
batch_norm_epsilon = 1e-5,
batch_norm_scale = True


def simple_dqnnet_bn(inputs, output_size=11, scope=None, is_training=True, reuse=None, regularizer=0.0,
                     batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with tf.variable_scope(scope, 'net', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection,
                            activation_fn=None,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm],**batch_norm_params, is_training=is_training):
                net = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='bn_preact')
                net = slim.conv2d(net, 32, 8, [4, 4], scope='conv1', padding='SAME')
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn_conv1')
                net = slim.conv2d(net, 64, 4, [2, 2], scope='conv2', padding='SAME')
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn_conv2')
                conv_out = slim.flatten(slim.conv2d(net, 64, 3, [1, 1], scope='conv3', padding='SAME'), scope='flatten')
                net = slim.batch_norm(conv_out, activation_fn=tf.nn.relu, scope='bn_conv3')
                net = slim.fully_connected(net, 512, scope='fc1')
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn_fc1')
                prediction = slim.fully_connected(net, output_size, activation_fn=None, scope='fc2')

                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return prediction, end_points



def simple_dqnnet_irr(inputs,current_irradiance, output_size=11, scope=None, is_training=True, reuse=None, regularizer=0.0):
    with tf.variable_scope(scope, 'net', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'


        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection,
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(inputs, 32, 8, [4, 4], scope='conv1', padding='SAME')
            net = slim.conv2d(net, 64, 4, [2, 2], scope='conv2', padding='SAME')
            conv_out = slim.flatten(slim.conv2d(net, 64, 3, [1, 1], scope='conv3', padding='SAME'), scope='flatten')
            net = slim.fully_connected(conv_out, 512, scope='fc1')

            net = tf.concat([net, current_irradiance], axis=1)
            print("NET_SHAPE",net.get_shape())
            prediction = slim.fully_connected(net, output_size, activation_fn=None, scope='fc2')

            biased_prediction = tf.add(prediction,current_irradiance)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return biased_prediction, end_points


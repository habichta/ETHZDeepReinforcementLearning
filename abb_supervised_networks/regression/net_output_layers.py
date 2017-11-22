
import tensorflow as tf
slim = tf.contrib.slim







def fc2084_fc_2084_lin1(model_output, regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())

    if is_training:
        with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                weights_initializer=slim.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(regularizer),
                                outputs_collections=end_points_collection):
                net = slim.fully_connected(model_output, 2048, scope='fc1')
                #net = slim.dropout(net, 0.5, scope='dropout1')
                net = slim.fully_connected(net, 2048, scope='fc2')
                #net = slim.dropout(net, 0.5, scope='dropout2')
                prediction = slim.fully_connected(net, 1, activation_fn=None, scope='fc3')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return prediction, end_points
    else:

        with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                                weights_initializer=slim.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(regularizer),
                                outputs_collections=end_points_collection):
                net = slim.fully_connected(model_output, 2048, scope='fc1')
                net = slim.fully_connected(net, 2048, scope='fc2')
                prediction = slim.fully_connected(net, 1, activation_fn=None, scope='fc3')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                return prediction, end_points



def linear(model_output, regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())


    with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection):
            prediction = slim.fully_connected(model_output, 1, activation_fn=None, scope='fc3')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return prediction, end_points


def linear11(model_output, regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())


    with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection):
            prediction = slim.fully_connected(model_output, 11, activation_fn=None, scope='fc3')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return prediction, end_points


def linear11_irr(model_output, current_irradiance,regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())


    with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection):
            prediction = slim.fully_connected(model_output, 11, activation_fn=None, scope='fc3')
            biased_prediction = tf.add(prediction, current_irradiance)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return biased_prediction, end_points

def linear31_irr(model_output, current_irradiance,regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())


    with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection):
            prediction = slim.fully_connected(model_output, 31, activation_fn=None, scope='fc3')
            biased_prediction = tf.add(prediction, current_irradiance)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return biased_prediction, end_points



def linear11_dout(model_output, regularizer = 0.0 , is_training = True, scope='output_layer', reuse=None):
    print('RESNET_OUTPUTSHAPE: ', model_output.get_shape())


    if is_training:
        keep_prob = 0.5
    else:
        keep_prob = 1.0

    with tf.variable_scope(scope, [model_output], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularizer),
                            outputs_collections=end_points_collection):
            net = slim.fully_connected(model_output, 11, activation_fn=None, scope='fc3')
            prediction = slim.dropout(net, keep_prob, scope='dropout1')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return prediction, end_points

def identity(model_input, regularization = 0.0, is_training=True, scope='output_layer_identity', reuse=None):
    end_points = {}
    return model_input, end_points


output_layer_factory = {"fc2084_fc_2084_lin1":fc2084_fc_2084_lin1,
                        "linear": linear,
                        "linear11": linear11,
                        "linear11_irr": linear11_irr,
                        "linear31_irr": linear31_irr,
                        "linear11_dout": linear11_dout,
                        "identity":identity}




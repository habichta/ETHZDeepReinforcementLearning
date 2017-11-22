
import tensorflow as tf
slim = tf.contrib.slim





def conv_6_to_3_10_10(model_input, regularization = 0.0, is_training=True, scope='input_layer_conv_6_to_3', reuse=None):
    print('MODEL_INPUT_SHAPE: ', model_input.get_shape())

    with tf.variable_scope(scope, [model_input], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(regularization),
                            outputs_collections=end_points_collection):
            corr_image_patch = slim.conv2d(model_input, 3, [10, 10], scope='correlation_layer')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return corr_image_patch, end_points



def identity(model_input, regularization = 0.0, is_training=True, scope='input_layer_identity', reuse=None):
    print('MODEL_INPUT_SHAPE: ', model_input.get_shape())
    end_points = {}

    return model_input, end_points

input_layer_factory = {"conv_6_to_3_10_10":conv_6_to_3_10_10,
                       "identity":identity}




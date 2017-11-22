import tensorflow as tf

slim = tf.contrib.slim
import net_resnetV2
import net_simple_dqnnet




def regression_resnetV250(model_input, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_50(inputs=model_input, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_50',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze)
    return output, end_points_resnet


def regression_resnetV218(model_input, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_18(inputs=model_input, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_18',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              include_root_block=include_root_block)
    return output, end_points_resnet

def regression_resnetV218_bottleneck(model_input, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_18_bottleneck(inputs=model_input, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_18',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              include_root_block=include_root_block)
    return output, end_points_resnet

def regression_resnetV218_irr(model_input, current_irradiance, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, pooling_layer = True,include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_18_irr(inputs=model_input,current_irradiance=current_irradiance, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_18',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              pooling_layer=pooling_layer,
                                                              include_root_block=include_root_block)
    return output, end_points_resnet




def regression_resnetV218_irr_bottleneck(model_input, current_irradiance, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, pooling_layer = True,include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_18_irr_bottleneck(inputs=model_input,current_irradiance=current_irradiance, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_18',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              spatial_squeeze=spatial_squeeze,
                                                              pooling_layer=pooling_layer,
                                                              include_root_block=include_root_block)
    return output, end_points_resnet


def regression_resnetV250_irr(model_input, current_irradiance, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                          global_pool, output_stride, spatial_squeeze, pooling_layer=True,include_root_block=True, is_training=True,
                          reuse=None):
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_50_irr(inputs=model_input,current_irradiance=current_irradiance, num_classes=None,
                                                              is_training=is_training, scope='resnet_v2_50',
                                                              global_pool=global_pool,
                                                              output_stride=output_stride,
                                                              pooling_layer=pooling_layer,
                                                              spatial_squeeze=spatial_squeeze)
    return output, end_points_resnet




def regression_resnetV218_nopool(model_input, weight_decay, batch_norm_decay, batch_norm_epsilon, batch_norm_scale,
                                 global_pool, output_stride, spatial_squeeze, is_training=True, reuse=None):
    """
    Removes the pooling layer and increases stride  from 2 to 4  in first convolutional layer and also increases kernel size from 7 to 8
    """
    with slim.arg_scope(
            net_resnetV2.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=batch_norm_decay,
                                          batch_norm_epsilon=batch_norm_epsilon,
                                          batch_norm_scale=batch_norm_scale)):
        output, end_points_resnet = net_resnetV2.resnet_v2_18_nopool(inputs=model_input, num_classes=None,
                                                                     is_training=is_training,
                                                                     scope='resnet_v2_18_nopool',
                                                                     global_pool=global_pool,
                                                                     output_stride=output_stride,
                                                                     spatial_squeeze=spatial_squeeze,
                                                                     )
    return output, end_points_resnet


def regression_simple_dqnnet(model_input, output_size, weight_decay, is_training=True, reuse=None):
    output, end_points = net_simple_dqnnet.simple_dqnnet(inputs=model_input, output_size=output_size,
                                                         scope="simple_dqnnet", is_training=is_training, reuse=reuse,
                                                         regularizer=weight_decay)
    return output, end_points



def regression_simple_dqnnet_do(model_input, output_size, keep_prob, weight_decay, is_training=True, reuse=None):
    output, end_points = net_simple_dqnnet.simple_dqnnet_do(inputs=model_input, output_size=output_size,
                                                         keep_prob=keep_prob, scope="simple_dqnnet",
                                                         is_training=is_training, reuse=reuse, regularizer=weight_decay)
    return output, end_points


def regression_simple_dqnnet_bn(model_input, output_size, weight_decay, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
                                batch_norm_scale=True, is_training=True, reuse=None):
    output, end_points = net_simple_dqnnet.simple_dqnnet_bn(inputs=model_input, output_size=output_size,
                                                         scope="simple_dqnnet", is_training=is_training, reuse=reuse,
                                                         regularizer=weight_decay, batch_norm_decay=batch_norm_decay,
                                                         batch_norm_epsilon=batch_norm_epsilon,
                                                         batch_norm_scale=batch_norm_decay)
    return output, end_points


def regression_simple_dqnnet_irr(image_inputs,current_irradiance, output_size,  weight_decay, is_training=True, reuse=None,):
    output, end_points = net_simple_dqnnet.simple_dqnnet_irr(inputs=image_inputs,current_irradiance=current_irradiance, output_size=output_size,
                                                         scope="simple_dqnnet", is_training=is_training, reuse=reuse,
                                                         regularizer=weight_decay)
    return output, end_points

network_factory = {"regression_resnetV250": regression_resnetV250,
                   "regression_resnetV250_irr":regression_resnetV250_irr,
                   "regression_resnetV218": regression_resnetV218,
                   "regression_resnetV218_bottleneck": regression_resnetV218_bottleneck,
                   "regression_resnetV218_irr": regression_resnetV218_irr,
                   "regression_resnetV218_irr_bottleneck": regression_resnetV218_irr_bottleneck,
                   "regression_resnetV218_nopool": regression_resnetV218_nopool,
                   "regression_simple_dqnnet": regression_simple_dqnnet,
                   "regression_simple_dqnnet_do": regression_simple_dqnnet_do,
                   "regression_simple_dqnnet_bn": regression_simple_dqnnet_bn,
                   "regression_simple_dqnnet_irr":  regression_simple_dqnnet_irr
                   }

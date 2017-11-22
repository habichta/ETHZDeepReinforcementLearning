'''
Created on Mon Mar20 10:12:24 2017

Residual Network Builder module

'''
from keras.layers.core import Lambda

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.layers.merge import (add,concatenate)

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf


def _bn_relu(input):
    """BN -> relu block
    """
    norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        conv = Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample,
                             kernel_initializer=init, padding=border_mode, kernel_regularizer=W_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """BN -> relu -> conv block.
    """
    nb_filter = conv_params["nb_filter"]
    nb_row = conv_params["nb_row"]
    nb_col = conv_params["nb_col"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row,nb_row), strides=subsample,
                             kernel_initializer=init, padding=border_mode, kernel_regularizer=W_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(filters=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_size=(1,1),
                                 strides=(stride_width, stride_height),
                                 kernel_initializer="he_normal", padding="valid",
                                 kernel_regularizer=l2(0.0001))(input)

    return merge([shortcut, residual], mode="sum")


def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Convolution2D(filters=nb_filter,
                                 kernel_size=(3,3),
                                 strides=init_subsample,
                                 kernel_initializer="he_normal", padding="same",
                                 kernel_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    :return: A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            conv_1_1 = Convolution2D(nb_filter=nb_filter,
                                 nb_row=1, nb_col=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, nb_row=1, nb_col=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, nb_row=3, nb_col=3)(conv_1_1)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, nb_row=1, nb_col=1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


class ResnetBuilder(object):
    @staticmethod
    def build(img_input_shape,irr_input_shape, num_outputs, block_fn, repetitions,input_irradiance):



        handle_dim_ordering()
        if len(img_input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            img_input_shape = (img_input_shape[1], img_input_shape[2], img_input_shape[0])

        img_input = Input(shape=img_input_shape,name="img_input")
        irr_input = Input(shape=irr_input_shape,name="irr_input")


        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(img_input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) #TODO: ev. remove

        block = pool1
        nb_filter = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # Last activation
        block = _bn_relu(block)

        block_norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
                                            block._keras_shape[COL_AXIS]),
                                 strides=(1, 1))(block_output)
        if input_irradiance:
            flatten1 = Flatten()(pool2)
            concat = concatenate([flatten1,irr_input])

            d = Dense(units=num_outputs, kernel_initializer="he_normal", activation="linear",
                          name="regression_output_irr")(concat)
            multOut = Lambda(lambda x: x[0] + x[1],output_shape=(num_outputs,))([d,irr_input])           
            dense = multOut
        else:
            flatten1 = Flatten()(pool2)

            dense = Dense(units=num_outputs, kernel_initializer="he_normal", activation="linear",name="regression_output")(flatten1)

        if input_irradiance:
            model = Model(inputs=[img_input,irr_input], outputs=dense)
        else:
            model = Model(inputs=[img_input], outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(img_input_shape,irr_input_shape, num_outputs,input_irradiance):
        return ResnetBuilder.build(img_input_shape, irr_input_shape, num_outputs, basic_block, [2, 2, 2, 2],input_irradiance)

    """
    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

    """

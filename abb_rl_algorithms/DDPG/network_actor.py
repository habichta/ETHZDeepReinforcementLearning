import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


class Qnetwork():
    def __init__(self, environment, img_size=84, img_sequence_len=2,img_channels=3, stream_hidden_layer_size=256,huber_delta=1.0):
        """
        class to construct a DQN
        :param hidden_layer_size: size of embedding before
        """

        self.input_image_sequence = tf.placeholder(shape=[None,img_size,img_size,img_sequence_len*img_channels],dtype=tf.float32)
        self.input_current_irradiance = tf.placeholder(shape=[None, img_sequence_len], dtype=tf.float32) # all previous irradiance values in sequence
        self.input_current_control_input = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.env = environment
        self.huber_delta=huber_delta

        #Hidden layers
        self.stream_hidden_layer_size = stream_hidden_layer_size

    def simple_dqn(self, regularizer, scope='simple_dqn', reuse=None):

        with tf.variable_scope(scope, 'net', [self.input_image_sequence], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=slim.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(regularizer),
                                outputs_collections=end_points_collection,
                                activation_fn=tf.nn.relu):
                self.conv1 = slim.conv2d(self.input_image_sequence, 32, 8, [4, 4], scope='conv1', padding='VALID')
                self.conv2 = slim.conv2d(self.conv1, 64, 4, [2, 2], scope='conv2', padding='VALID')
                self.conv3 = slim.conv2d(self.conv2, 64, 3, [1, 1], scope='conv3', padding='VALID')

                self.conv_out = slim.flatten(inputs=self.conv3, scope='flatten')

                # Critic gets action from Actor network and concatenates it.

                self.concat_layer = tf.concat([self.conv_out, self.input_current_irradiance], axis=1)

                self.fc1 = slim.fully_connected(self.concat_layer, self.stream_hidden_layer_size, scope='fc1')

                self.unscaled_action_output = slim.fully_connected(self.fc1, 1, activation_fn=tf.nn.tanh, scope='fc2') #outut action between [-1,1] need to be scaled later

                # TRAINING:
                self.learning_rate = tf.placeholder(dtype=tf.float32)
                #Input given by critic network
                self.action_gradient = tf.placeholder(shape=[None,1],dtype=tf.float32) #Assume one-dim action space
                self.action_bounds = tf.placeholder(shape=[None,1],dtype=tf.float32) #enforce the ramps, since each experience has slightly different, these need to be calculated for each iteration

                self.scaled_action_input = tf.multiply(self.unscaled_action_output,self.action_bounds)

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope)

                self.actor_gradient = tf.gradients(self.scaled_action_input,self.trainable_variables,-self.action_gradient)

                #TODO print action gradients, why action gradients[0]? divide by batchsize?

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.updateModel = self.optimizer.apply_gradients(zip(self.actor_gradient,self.trainable_variables))
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)


        with tf.variable_scope(scope, reuse=True):
            self.first_layer_weights = tf.get_variable("conv1/weights")

    #Huberloss to prevent outliers from creating too large gradients. Otherwise use MSE loss
    def td_huber_loss(self,td_error, delta=1.0): #reward should be around 0 to -1000 so MSE in that range, but not above ... maybe clip reward to 0 to -1000 as well... #TODO: reward clipping
        """
        The authors of the paper employ gradient clipping (not loss clipping), this can be achieved with the huber loss
        since its gradient will become
        :param targetQ: target network Q values
        :param Q: training network Q values
        :param delta: threshhold where  after which error gradient will be 1 or -1
        :return: loss. notice that gradient will be clipped..
        """
        residual = tf.abs(td_error)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)

    @staticmethod
    def target_update_operations(trainable_variables, tau):
        """
        :param trainable_variables: contains the variables of both network, so divide num_trainable_variables by 2
        :param tau: how fast to update target network with training network
        :return: list of operations to perform in a session
        """
        num_trainable_variables = len(trainable_variables)
        assign_operations = []
        for idx, var in enumerate(trainable_variables[0:num_trainable_variables // 2]): #go through var of training network
            assign_operations.append(trainable_variables[idx + num_trainable_variables // 2].assign( #assign update to target variables (second half of all the variables)
                (var.value() * tau) + (
                (1 - tau) * trainable_variables[idx + num_trainable_variables // 2].value())))

        return assign_operations

    @staticmethod
    def update_target_network(assign_operations,sess):
        for a_op in assign_operations:
            sess.run(a_op)
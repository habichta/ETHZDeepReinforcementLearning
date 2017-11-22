import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


class Critic_Network():
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

                #Critic gets action from Actor network and concatenates it. Also gets irradiance history of current state
                self.action_input = tf.placeholder(shape=[None], dtype=tf.float32)
                self.concat_layer = tf.concat([self.conv_out,self.action_input,self.input_current_irradiance],axis=1)


                self.fc1 = slim.fully_connected(self.concat_layer, self.stream_hidden_layer_size, scope='fc1')

                self.Q = slim.fully_connected(self.fc1,1,activation_fn=None,scope='fc2')


                #TRAINING:
                #targetQ will be y = r + gamma*Q'(s_t+1,mu'(s_s+1|theta_actor')|theta_critic')
                self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)  # input is a batch of targetQ values

                self.learning_rate = tf.placeholder(dtype=tf.float32) #dynamic learning rate input


                self.td_error = self.Q-tf.stop_gradient(self.targetQ) #stop_gradient is redundant
                self.batch_losses = self.td_huber_loss(self.td_error,delta=self.huber_delta)
                self.loss = tf.reduce_mean(self.batch_losses)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.gradients = self.optimizer.compute_gradients(self.loss)
                self.updateModel = self.optimizer.apply_gradients(self.gradients)
                self.end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                #Get the credit assignment of the action in relation to Q value approximation. Used for gradient in actor network
                self.action_gradient = self.gradients(self.Q,self.action_input) #batch of action inputs


        with tf.variable_scope(scope,reuse=True):
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



    #TODO: NON STATIC, for each network type ...

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
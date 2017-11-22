import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


# layer_norm: https://arxiv.org/abs/1607.06450

class Actor(Model):
    def __init__(self, network, nr_actions, name='actor_network', layer_norm=True, l2_regularizer=0.0, hidden_units=256):
        super(Actor, self).__init__(name=name)
        self.nr_actions = nr_actions
        self.layer_norm = layer_norm
        self.l2_regularizer = l2_regularizer
        self.hidden_units = hidden_units
        self.network = network #network function (partial function) partial(network, all_parametrers except input)

    def __call__(self, input_data, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = self.network(input_data)

            return output

    def dqn_network(self, input_data,hidden_units=256):

        x = input_data
        #TODO handle input_data correctly

        for i, t in enumerate([(32, 8, 4), (32, 4, 2), (64, 3, 1)]):
            x = tf.layers.conv2d(x, t[0], t[1], (t[2], t[2]), name="conv{}".format(i+1), padding="valid",
                                 activation=tf.nn.relu, kernel_initializer=tc.layers.xavier_initializer_conv2d,
                                 kernel_regularizer=tc.layers.l2_regularizer(self.l2_regularizer))

        x = tf.layers.dense(x, hidden_units,kernel_regularizer=tc.layers.xavier_initializer())
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, self.nr_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        x = tf.nn.tanh(x)

        return x



class Critic(Model):
    def __init__(self,network,name="critic_network",layer_norm=True,l2_regularizer=0.0,hidden_units=256):
        super(Critic,self).__init__(name=name)
        self.layer_norm = layer_norm
        self.l2_regularizer = l2_regularizer
        self.hidden_units = hidden_units
        self.network = network

    def __call__(self, input_data, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            output = self.network(input_data,action)

            return output


    def dqn_network(self, input_data, action,hidden_units=256):

        x = input_data

        for i, t in enumerate([(32, 8, 4), (32, 4, 2), (64, 3, 1)]):
            x = tf.layers.conv2d(x, t[0], t[1], (t[2], t[2]), name="conv{}".format(i+1), padding="valid",
                                 activation=tf.nn.relu, kernel_initializer=tc.layers.xavier_initializer_conv2d,
                                 kernel_regularizer=tc.layers.l2_regularizer(self.l2_regularizer))

        x = tf.layers.dense(x, hidden_units,kernel_regularizer=tc.layers.xavier_initializer())
        if self.layer_norm:
            x = tc.layers.layer_norm(x, center=True, scale=True)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
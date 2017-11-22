from copy import copy
import tensorflow as tf
import numpy as np


def get_target_updates(vars, target_vars, tau):
    soft_updates = []
    init_updates = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(vars)
    assert len(soft_updates) == len(vars)
    return tf.group(*init_updates), tf.group(*soft_updates)



class DDPG(object):
    def __init__(self, actor, critic, replay_buffer, input_images_shape,input_irradiance_shape,input_control_shape,action_shape,action_noise=None, gamma=0.99, tau=0.00001, batch_size=32, action_range=[-1.,1.],
               critic_l2_reg=0.0, actor_l2_reg=0.0, critic_lr=1e-3, actor_lr=1e-4):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = replay_buffer
        self.action_noise = action_noise
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.actor_l2_reg = actor_l2_reg
        self.action_range = action_range

        self.input_images_shape = input_images_shape
        self.input_irradiance_shape = input_irradiance_shape
        self.input_control_shape = input_control_shape

        #####################
        # TARGET NETWORKS
        #####################

        target_actor = copy(actor)
        target_actor.name = 'target_actor_network'
        target_critic = copy(critic)
        target_critic.name = 'target_critic_network'

        self.target_actor = target_actor
        self.target_critic = target_critic


        #####################
        #TF OPS FOR DDPG
        #####################

        # input_data_current state #TODO: give whole experience as input and decode here
        self.input_images_0_tf = tf.placeholder(tf.float32, shape=(None,) + input_images_shape, name="input_images0")
        self.input_irradiance_0_tf = tf.placeholder(tf.float32, shape=(None,) + input_irradiance_shape, name="input_irradiance0")
        self.input_control_0_tf = tf.placeholder(tf.float32, shape=(None,) + input_control_shape, name="input_control0")
        #next state:
        self.input_images_1_tf = tf.placeholder(tf.float32, shape=(None,) + input_images_shape, name="input_images1")
        self.input_irradiance_1_tf = tf.placeholder(tf.float32, shape=(None,) + input_irradiance_shape,
                                                  name="input_irradiance1")
        self.input_control_1_tf = tf.placeholder(tf.float32, shape=(None,) + input_control_shape, name="input_control1")

        #rewards and done
        self.rewards_tf = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions_tf = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self.done_s1_tf = tf.placeholder(tf.float32, shape=(None, 1), name='done_s1')



        input_data0 = [self.input_images_0_tf,self.input_irradiance_0_tf,self.input_control_0_tf]
        input_data1 = [self.input_images_1_tf, self.input_irradiance_1_tf, self.input_control_1_tf]

        #NETWORK_OUTPUTS
        self.actor_output_tf = actor(input_data0)
        self.critic_output_tf = critic(input_data0,self.actions_tf) #used for updating the critic given batch actions
        self.critic_with_actor_output_tf = critic(input_data0,self.actor_output_tf,reuse=True) #update actor, the actor creates new action and gets score by updated critic

        #TARGET_Q
        self.Q1_tf = target_critic(input_data1,target_actor(input_data1)) #use target networks to get targetQ
        self.target_Q_tf = self.rewards_tf + (1.0 -self.done_s1_tf)*gamma*self.Q1_tf

        self.setup_optimizer()
        self.setup_target_network_updates()


    def setup_optimizer(self):
        self.critic_target_tf = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        self.critic_loss_tf = tf.reduce_mean(tf.square(self.critic_output_tf - self.critic_target_tf))
        self.critic_gradients_tf = tf.gradients(self.critic_loss_tf, self.critic.trainable_vars)
        self.critic_optimizer_tf = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
        self.update_critic_tf = self.critic_optimizer_tf.apply_gradients(self.critic_gradients_tf)

        with tf.control_dependencies([self.update_critic_tf]):
            self.actor_loss_tf = -tf.reduce_mean(self.critic_with_actor_output_tf)
            self.actor_gradients_tf = tf.gradients(self.actor_loss_tf, self.actor.trainable_vars)
            self.actor_optimizer_tf = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
            self.update_actor_tf = self.actor_optimizer_tf.apply_gradients(self.actor_gradients_tf)





    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars,
                                                                      self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_updates)  # copy over var values to target


    def store_experience(self,experience,td_error=1):

        self.replay_buffer.add(td_error,experience)

    def sample_experience(self):

        return self.replay_buffer.sample(self.batch_size,None)

    def update_experience(self,idx, td_error):
        self.replay_buffer.update(idx, td_error)

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_train_feed_dict(self,training_batch):
        input_images_batch_0 = np.stack(np.stack(training_batch[:, 0])[..., 0])
        input_irradiance_batch_0 = np.stack(np.stack(training_batch[:, 0])[..., 1])
        input_control_batch_0 = np.reshape(np.stack(np.stack(training_batch[:, 0])[..., 2]),
                                           [-1, 1])

        input_images_batch_1 = np.stack(np.stack(training_batch[:, 3])[..., 0])
        input_irradiance_batch_1 = np.stack(np.stack(training_batch[:, 3])[..., 1])
        input_control_batch_1 = np.reshape(np.stack(np.stack(training_batch[:, 3])[..., 2]), [-1, 1])

        reward_batch = training_batch[:, 2]
        action_batch = training_batch[:, 1]
        done_batch = training_batch[:, 4]

        feed_dict={self.input_images_0_tf:input_images_batch_0,
                   self.input_irradiance_0_tf:input_irradiance_batch_0,
                   self.input_control_0_tf:input_control_batch_0,
                   self.input_images_1_tf:input_images_batch_1,
                   self.input_irradiance_1_tf:input_irradiance_batch_1,
                   self.input_control_1_tf:input_control_batch_1,
                   self.rewards_tf:reward_batch,
                   self.actions_tf:action_batch,
                   self.done_s1_tf:done_batch
                   }

        return feed_dict

    def get_predict_feed_dict(self,state):

        input_images = np.array([state[0]])
        input_irradiance = np.reshape(state[1], self.input_irradiance_shape)
        input_control = np.reshape(state[2], self.input_control_shape)[0]

        feed_dict = {self.input_images_0_tf:input_images,
                     self.input_irradiance_0_tf:input_irradiance,
                     self.input_control_0_tf:input_control}
        return feed_dict


    def train_step(self):

        training_batch = self.sample_experience()
        feed_dict = self.get_train_feed_dict(training_batch)
        targetQ = self.sess.run(self.target_Q_tf,feed_dict=feed_dict)
        feed_dict = feed_dict+{self.target_Q_tf:targetQ}

        _, critic_loss = self.sess.run([self.update_critic_tf,self.critic_loss_tf], feed_dict=feed_dict)
        _, actor_loss = self.sess.run([self.update_actor_tf, self.actor_loss_tf], feed_dict=feed_dict)


        return critic_loss,actor_loss


    def predict_action(self,state,apply_noise=True,compute_Q=True):

        feed_dict = self.get_predict_feed_dict(state)

        if compute_Q:
            action,Q = self.sess.run([self.actor_output_tf,self.critic_with_actor_output_tf],feed_dict=feed_dict)
        else:
            action = self.sess.run(self.actor_output_tf, feed_dict=feed_dict)
            Q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise

        action = np.clip(action,self.action_range[0],self.action_range[1])

        return action,Q



    def reset(self):
        if self.action_noise is not None:
            self.action_noise.reset()


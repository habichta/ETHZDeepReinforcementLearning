from abb_rl_algorithms.DDPG.DDPG import DDPG
import tensorflow as tf
from collections import deque
import time

def load_checkpoint(sess,restorer,path):
    if path and restorer:
        restorer.restore(sess,path)

def create_train_writer(sess,train_dir):
    return tf.summary.FileWriter(train_dir,
                                         sess.graph)


def train(train_dir,env,nr_epochs, pre_train_steps,update_frequency,actor, critic, replay_buffer, input_images_shape,input_irradiance_shape,input_control_shape,action_shape,action_noise=None, gamma=0.99, tau=0.00001, batch_size=32, action_range=[-1.,1.],
               critic_l2_reg=0.0, actor_l2_reg=0.0, critic_lr=1e-3, actor_lr=1e-4,pretrained_checkpoint_path=None,restorer=None):

    agent = DDPG(actor, critic, replay_buffer, input_images_shape,input_irradiance_shape,input_control_shape,action_shape,action_noise, gamma, tau, batch_size, action_range,
               critic_l2_reg, actor_l2_reg, critic_lr, actor_lr)





    with tf.Session as sess:

        agent.initialize(sess)
        sess.graph.finalize()
        agent.reset()

        load_checkpoint(sess=sess,restorer=restorer,path=pretrained_checkpoint_path)
        train_writer = create_train_writer(sess,train_dir)


        #Initialize
        start_time = time.time()
        total_steps = 0


        for epoch_i in nr_epochs:

            epoch_reward_list = list()
            epoch_q_value_list = list()



            for episode_i in range(len(train_epoch_list)):

                done = 0
                episode_reward_list = list()
                episode_mean_q_value_list = list()
                episode_steps = 0
                mean_max_q_value = 0
                total_loss = 0
                state, abort = env.reset()

                #Episode Starts
                while done == 0 and not abort:

                    episode_steps+=1
                    total_steps+=1

                    action, Q = agent.predict_action(state,apply_noise=True,compute_Q=True)

                    next_state,reward,done = env.step(action) #TODO calculation of next step, change environment actions

                    experience = [state,action,reward,next_state,done]

                    agent.store_experience(experience,td_error=1.0)


                    if total_steps > pre_train_steps and total_steps % update_frequency == 0:
                        critic_loss, actor_loss = agent.train_step()

                    #Todo how to to pretraining, train critic alone?
                    #reward scaling?
                    #How to scale actions ... fixed step? just multiply by maximum ramp?








                    state=next_state







        #TODO: pretrain?
        #TODO: how change action noise
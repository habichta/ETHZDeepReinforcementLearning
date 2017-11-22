import tensorflow as tf
import os, csv
import datetime as dt
from network import Qnetwork
from environment import Environment
from experience_replay_buffer import ExperienceReplayBuffer
import numpy as np
import time
import rl_logging
import collections
import matplotlib.pyplot as plt
import random

slim = tf.contrib.slim
from scipy import misc
import pandas as pd

FLAGS = tf.app.flags.FLAGS

################################
# General
################################

np.random.seed(1337) #Reproducibility
random.seed(1337)
tf.set_random_seed(1337)


tf.app.flags.DEFINE_float(
    'per_process_gpu_memory_fraction', 0.3,
    'fraction of gpu memory used for this process')
#run_simple_irr_dqn_hl_mse_tau_0001_e_200_followirr_randomexp_sunspot_action_naive_nopt_mpc_divi_mask_uf1/
tf.app.flags.DEFINE_string(
    'train_dir',
    '/home/dladmin/Documents/arthurma/experiments_rl/baseline/run_simple_duelling_dqn_hl_300_tau_0001_e_200_followirr_randomexp_sunspot_action_naive_nopt_mpc_divi_mask_uf1_addirr/',
    'Directory where checkpoints, info, and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrained_checkpoint_path',
    None,
    'The path to a checkpoint from which to fine-tune. Only if restore_latest_checkpoint is false')

tf.app.flags.DEFINE_boolean(
    'use_restore_dict',
    False,
    'For partial restore,update dictionary in network.py')

tf.app.flags.DEFINE_string(
    'train_set_path',
    '/home/dladmin/Documents/arthurma/shared/dlabb/abb_rl_algorithms/DDDQN/train_list.out',
    'Directory where File with train set is')

tf.app.flags.DEFINE_integer(
    'sample_train_episodes',
    None,
    'Randomly sample a subset of training episodes')

tf.app.flags.DEFINE_string(
    'test_set_path',
    '/home/dladmin/Documents/arthurma/shared/dlabb/abb_rl_algorithms/DDDQN/validation_list.out',
    'Directory where File with test set is')

# /home/dladmin/Documents/arthurma/rf/eval/run_hl_mse_tau_00001_e_100_followirr_guidedexp_sunspot_per_50442_TRAIN/low_reward_episodes.pickle
#'/home/dladmin/Documents/arthurma/rf/low_reward_episodes200.pickle'
tf.app.flags.DEFINE_string(
    'load_train_episodes'
    ,"/home/dladmin/Documents/arthurma/rf/low_reward_episodes200.pickle",
    'Directory where File with test set is')

tf.app.flags.DEFINE_string(
    'load_test_episodes',
    None,
    'Directory where File with test set is')

tf.app.flags.DEFINE_string(
    'data_file',
    'rl_data_sp.csv',
    'rl_data_sp.csv for images with green spot over sun')

################################
# Input Data
################################
tf.app.flags.DEFINE_integer(
    'batch_size', 128,
    'samples from experience replay, randomly sampled')

tf.app.flags.DEFINE_integer(
    'img_size', 84,
    'height/width of input image')

tf.app.flags.DEFINE_integer(
    'img_sequence_length', 2,
    'nr of image in one sequence')

tf.app.flags.DEFINE_integer(
    'img_sequence_stride', 9,
    'nr of image in one sequence')

tf.app.flags.DEFINE_float(
    'divide_image_values', 255.0,
    '255.0 or 1.0 or None (255 -> values between 0 and 1) Note: converts values to float32, increases memory usage by 4')

tf.app.flags.DEFINE_string(
    'mask', '/media/data/Daten/img_C/cavriglia_skymask256.png',
    'Mask applied to images')

################################
# Simple QNetwork
################################

tf.app.flags.DEFINE_string(
    'network', "simple_duelling_dqn",
    'simple_irr_dqn, simple_duelling_dqn')

tf.app.flags.DEFINE_boolean(
    'add_irr', True,
    'Add current irradiance to the model')

tf.app.flags.DEFINE_float(
    'stream_hidden_layer_size', 256,
    'hidden layer size before Q-value regression')

################################
# Training
################################


tf.app.flags.DEFINE_float(
    'learning_rate', 0.00001,
    'Learning rate for gradient updates')

tf.app.flags.DEFINE_float(
    'huber_delta', 300.0,
    'Threshhold mse vs. abs loss')

tf.app.flags.DEFINE_integer(
    'num_epochs', 200,
    '')

tf.app.flags.DEFINE_integer(
    'episode_length_train', 200,
    '')

tf.app.flags.DEFINE_integer(
    'episode_length_test', None,
    '')

tf.app.flags.DEFINE_integer(
    'pre_train_steps', 65000,  # to small will lead to divergence because of non i i d samples
    'Number of random steps before training begins (Fill buffer)')

tf.app.flags.DEFINE_integer(
    'update_frequency', 1,
    'frequency of training steps while acting in the environment')
tf.app.flags.DEFINE_float(
    'discount_factor', 0.99,
    'Discount on target Q-values')

tf.app.flags.DEFINE_float(
    'tau', 0.0001,  # 0.00001 # too large will lead to divergence because target network follows main network too fast
    'Convergence parameter, how much of main network is copied to target network')

tf.app.flags.DEFINE_integer(
    'replay_buffer_size', 65000,
    'Max number of experiences in the experience replay buffer')

tf.app.flags.DEFINE_float(
    'l2_regularizer', 0.00004,
    'Regularization in training network')

tf.app.flags.DEFINE_float(
    'adam_epsilon', 0.1,
    'adam config')

################################
# Exploration Strategy
################################
tf.app.flags.DEFINE_float(
    'start_e_greedy', 1.0,
    'epsilon greedy exploration strategy (start probability for exploration)')
tf.app.flags.DEFINE_float(
    'end_e_greedy', 0.1,
    'epsilon greedy exploration strategy (start probability for exploration)')
tf.app.flags.DEFINE_integer(
    'annealing_steps', 1000,
    'How many training steps to reduce start_e to end_e')

tf.app.flags.DEFINE_float(
    'validation_end_e_greedy', 0.1,
    'fixed exploration probability for validation set')

tf.app.flags.DEFINE_float(
    'mpc_guided_exploration', None,
    'probability with which action is selected that follows MPC during exploration')

################################
# Action Space
################################
tf.app.flags.DEFINE_integer(
    'num_actions',1,
    'Nr of actions to choose from')

# Enforces ramp restrictions: 100.0: 10% per minute => environment calculates depending on the time change between states which is roughly 6-8 seconds
tf.app.flags.DEFINE_float(
    'max_ramp_change_per_minute', 100.0,
    'Assuming 3 actions: Stay put, change battery throughput up by max_ramp_change or down by -max_ramp_change')

tf.app.flags.DEFINE_boolean(
    'follow_irr_actions', True,
    'Assuming 3 actions: Action 0 tries to follow current irradiance if current input is not too far away, otherwise ramp towards irr')

################################
# Logging and Checkpoints
################################

tf.app.flags.DEFINE_integer(
    'validation_each_n_episodes',  # 782/152, 2178 in hard set(100), 3640 for 200 (1401 hard = 38%), 7206 for 100
    7280,
    'how often to go over validation set')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summaries_every_n_total_steps', 5000,
    'The frequency with which summaries are saved, in total_steps, None to switch off.')

tf.app.flags.DEFINE_integer(
    'save_checkpoint_after_n_episodes', 2802,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_nr_checkpoints_saved', 200,
    'Maximum number of newest checkpoints that are saved, older ones are deleted.')

tf.app.flags.DEFINE_boolean(
    'render_ep', False,
    'Render current train episode')


# nach gewisser anzahl episoden ...  / epochen  = nr. episodes ...


######################################################################################
##EPISODE RENDER
######################################################################################

def render_episode(current_control_input, current_episode):
    irr_list, mpc_list, ci_list, index_list = [], [], [], []
    column_list = ["irr", "mpc", "ci"]

    for t in current_control_input:
        index_list.append(t[1])
        ci_list.append(t[0])

    ci_df = pd.DataFrame(data=ci_list, index=index_list, columns=["ci"])

    irrmpc_df = current_episode.loc[ci_df.index]

    data_df = pd.concat([ci_df, irrmpc_df], axis=1)

    data_df[["ci", "irr", "mpc"]].plot()

    plt.show()


######################################################################################
##VALIDATION
######################################################################################

def do_validation_run(sess, network, env, train_writer, step):
    nr_validation_episodes = env.nr_test_episodes
    print("Validation run on " + str(nr_validation_episodes) + " episodes...")

    episodes_reward_list = list()
    episodes_q_value_list = list()
    chosen_action_list = list()

    for episode_nr in range(nr_validation_episodes):
        print("Validation Episode " + str(episode_nr + 1) + "/" + str(nr_validation_episodes))

        state, abort = env.test_reset()  # get next episode, initialize
        done = 0

        episode_reward_sum = 0
        episode_q_value_sum = 0
        episode_steps = 0

        while done == 0 and not abort:
            episode_steps += 1

            action, action_value_qs = \
                sess.run([network.predict, network.Qout],
                         feed_dict={network.input_image_sequence: (np.array([state[0]])),
                                    network.input_current_irradiance: np.reshape(
                                        [state[1]],
                                        [-1, FLAGS.img_sequence_length]),
                                    network.input_current_control_input: np.reshape(
                                        [state[2]],
                                        [-1, 1])})

            chosen_action_list.append(action[0])

            mean_max_action_value_q = np.mean(np.max(action_value_qs, axis=1))

            next_state, reward, done = env.test_step(action)

            episode_reward_sum += reward
            episode_q_value_sum += mean_max_action_value_q

            state = next_state

        print("Cumulative episode " + str(episode_nr + 1) + " reward:", episode_reward_sum)
        print("Average episode " + str(episode_nr + 1) + " reward:", episode_reward_sum / episode_steps)
        print("Average episode " + str(episode_nr + 1) + " Q Value:", episode_q_value_sum / episode_steps)
        episodes_reward_list.append(episode_reward_sum / episode_steps)
        episodes_q_value_list.append(episode_q_value_sum / episode_steps)

    action_counter = collections.Counter(chosen_action_list)

    rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=episodes_reward_list,
                               episodes_q_value_list=episodes_q_value_list, step=step, action_counter=action_counter,
                               set="validation_epoch")


#######################################################################################################################
##MAIN
#######################################################################################################################



def main(_):
    if tf.gfile.Exists(FLAGS.train_dir):
        raise ValueError("Train directory exists already")

    os.makedirs(FLAGS.train_dir)

    with open(os.path.join(FLAGS.train_dir, "train_model_info.csv"), "w+") as f:
        w = csv.writer(f)
        w.writerow([str(dt.datetime.now())])
        for key, val in FLAGS.__flags.items():
            w.writerow([key, val])

    ###################
    # Initialize
    ###################

    with tf.device('/gpu:0'):

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()

        # Environment and Networks
        env = Environment(train_set_path=FLAGS.train_set_path, test_set_path=FLAGS.test_set_path,
                          episode_length_train=FLAGS.episode_length_train, episode_length_test=FLAGS.episode_length_test,
                          sequence_length=FLAGS.img_sequence_length, sequence_stride=FLAGS.img_sequence_stride,
                          actions=FLAGS.num_actions,
                          image_size=FLAGS.img_size, follow_irr_actions=FLAGS.follow_irr_actions, file=FLAGS.data_file,
                          load_train_episodes=FLAGS.load_train_episodes, load_test_episodes=FLAGS.load_test_episodes,
                          mask_path=FLAGS.mask,
                          divide_image_values=FLAGS.divide_image_values,sample_training_episodes=FLAGS.sample_train_episodes)


        mainQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size, img_size=FLAGS.img_size,
                          img_sequence_len=FLAGS.img_sequence_length, huber_delta=FLAGS.huber_delta,epsilon=FLAGS.adam_epsilon,add_irr=FLAGS.add_irr)

        targetQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size,
                            img_size=FLAGS.img_size,
                            img_sequence_len=FLAGS.img_sequence_length, huber_delta=FLAGS.huber_delta,
                            epsilon=FLAGS.adam_epsilon,add_irr=FLAGS.add_irr)



        if FLAGS.network == "simple_duelling_dqn":
            mainQN.simple_duelling_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_main')
            targetQN.simple_duelling_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_target')
        elif FLAGS.network == "simple_irr_dqn":
            mainQN.simple_irr_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_irr_dqn_main')
            targetQN.simple_irr_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_irr_dqn_target')



        trainables = tf.trainable_variables()

        target_assign_operations = Qnetwork.target_update_operations(trainable_variables=trainables, tau=FLAGS.tau)
        target_full_assign_operations = Qnetwork.target_full_update_operations(trainable_variables=trainables)

        # Buffer to sample experience batches from (off-policy learning required)
        global_exp_buffer = ExperienceReplayBuffer(buffer_size=FLAGS.replay_buffer_size)

        # Initialize e_greedy exploration probability
        e = FLAGS.start_e_greedy
        e_reduction = (FLAGS.start_e_greedy - FLAGS.end_e_greedy) / FLAGS.annealing_steps

        # List of rewards and steps per episode
        epoch_reward_list = list()
        epoch_q_value_list = list()
        total_steps = 0
        num_episodes = env.nr_train_episodes * FLAGS.num_epochs  # how many episodes to run, each episode is visited before an episode is visited twice...
        num_episodes_per_epoch = env.nr_train_episodes

        tf_episode_nr_start = tf.get_variable("episode_counter", initializer=0)

        lr = FLAGS.learning_rate

        ###################
        # TRAINING SESSION
        ###################

        _, merge_op = rl_logging.create_summaries(mainQN, img_sequence_length=FLAGS.img_sequence_length)

        ###################
        # TRAINING SESSION
        ###################


        if FLAGS.use_restore_dict:
            with tf.variable_scope("simple_duelling_dqn_main", reuse=True):
                restore_dict = {"simple_dqnnet/conv1/weights": tf.get_variable('conv1/weights'),
                                "simple_dqnnet/conv2/weights": tf.get_variable('conv2/weights'),
                                "simple_dqnnet/conv3/weights": tf.get_variable('conv3/weights')}
                restorer = tf.train.Saver(var_list=restore_dict)
                print("Restore:", str(restore_dict))
        else:
            restorer = tf.train.Saver()

        saver = tf.train.Saver(max_to_keep=FLAGS.max_nr_checkpoints_saved)

        gradients = None  # for logging

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            Qnetwork.update_target_network(target_full_assign_operations,sess=sess) # initialize target network, setting it equal to main network



            if FLAGS.pretrained_checkpoint_path is not None:
                print('Loading pretrained model...')
                restorer.restore(sess, FLAGS.pretrained_checkpoint_path)

            # Tensorboard summary writer
            train_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                 sess.graph)

            start_time = time.time()

            # When checkpoints are loaded, continue from epoch/episode nr of checkpoint, this allows seamless graphs in tensorboard..
            first_episode = sess.run(tf_episode_nr_start)  # saver saves the finished episode, so start from next (+1)

            print("Starting from episode number:",
                  first_episode + 1)  # episode counting begins from 0 so +1 for natural counting

            for episode_nr in range(first_episode, num_episodes):  # go over all episodes in episode list:
                epoch = ((episode_nr + 1) // env.nr_train_episodes) + 1

                ###################
                # SAVE TRAIN EPOCH STATISTICS WHEN NEW EPOCH BEGINS
                ###################
                if (episode_nr + 1) % env.nr_train_episodes == 0:
                    if len(epoch_reward_list) > 0 and len(epoch_q_value_list) > 0:
                        print("Epoch finished: save training statistics")
                        rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=epoch_reward_list,
                                                   episodes_q_value_list=epoch_q_value_list, step=epoch,
                                                   set="training_epoch")
                    epoch_reward_list, epoch_q_value_list = list(), list()

                print("Epoch: ", str(epoch), "/", str(FLAGS.num_epochs))
                print("Episode number: ", str(episode_nr + 1))

                done = 0
                episode_reward_list = list()
                chosen_action_list = list()
                episode_mean_q_value_list = list()
                episode_steps = 0
                mean_max_q_value = 0
                total_loss = 0
                state, abort = env.reset()

                while done == 0 and not abort:  # within each episode:

                    episode_steps += 1
                    total_steps += 1

                    if (np.random.rand(1) < e):
                        # Explore
                        if FLAGS.mpc_guided_exploration is not None:
                            action = env.mpc_exploration(mpc_prob=FLAGS.mpc_guided_exploration,
                                                         num_actions=FLAGS.num_actions)
                        else:

                            action = np.random.randint(0, FLAGS.num_actions)

                    else:
                        # Predict Action
                        action = sess.run(mainQN.predict, feed_dict={mainQN.input_image_sequence: (np.array([state[0]])),
                                                                     mainQN.input_current_irradiance: np.reshape(state[1],
                                                                                                                 [-1,
                                                                                                                  FLAGS.img_sequence_length]),
                                                                     mainQN.input_current_control_input: np.reshape(
                                                                         state[2], [-1, 1])})[0]

                    next_state, reward, done = env.step(action)  # integer of action buffersize 50000

                    global_exp_buffer.add(np.reshape(np.array([state, action, reward, next_state, done]), [1, 5]))

                    if total_steps > FLAGS.pre_train_steps:  # create enough samples in experience replay buffer before training
                        if e > FLAGS.end_e_greedy:  # if end epsilon not reached, reduce e
                            e -= e_reduction

                        if total_steps % (FLAGS.update_frequency) == 0:
                            training_batch = global_exp_buffer.sample(FLAGS.batch_size, 5)

                            next_image_sequence_batch = np.stack(np.stack(training_batch[:, 3])[..., 0])
                            # misc.imshow(next_image_sequence_batch[6][:,:,0:3])
                            next_curr_irr_batch = np.stack(np.stack(training_batch[:, 3])[..., 1])
                            next_control_input_batch = np.reshape(np.stack(np.stack(training_batch[:, 3])[..., 2]), [-1, 1])

                            feed_dict_mqn = {mainQN.input_image_sequence: next_image_sequence_batch,
                                             mainQN.input_current_irradiance: next_curr_irr_batch,
                                             mainQN.input_current_control_input: next_control_input_batch}

                            feed_dict_tqn = {targetQN.input_image_sequence: next_image_sequence_batch,
                                             targetQN.input_current_irradiance: next_curr_irr_batch,
                                             targetQN.input_current_control_input: next_control_input_batch}

                            ### Calculate the target network value y (next state,next action maximum) (rhs of loss function)
                            Q1 = sess.run(mainQN.predict, feed_dict=feed_dict_mqn)  # select best action indices
                            Q2 = sess.run(targetQN.Qout,
                                          feed_dict=feed_dict_tqn)  # calculate all Q values of all action outputs in target network

                            end_multiplier = -(training_batch[:, 4] - 1)  # 1 if d = false  0 if d = True
                            doubleQ = Q2[range(  # Double Q Learning
                                FLAGS.batch_size), Q1]  # select the Q values of target network according to best action arguments of training network
                            # doubleQ => batchsize x 1  with Q values of best actions according to training network


                            targetQ = training_batch[:, 2] + (
                                FLAGS.discount_factor * doubleQ * end_multiplier)  # y = reward + disc_f*max(Q(s',a',theta'))

                            # Calculate Q(s,a,theta) and then get loss with targetQ as input, train on loss

                            next_train_image_sequence_batch = np.stack(np.stack(training_batch[:, 0])[..., 0])
                            next_train_curr_irr_batch = np.stack(np.stack(training_batch[:, 0])[..., 1])

                            next_train_control_input_batch = np.reshape(np.stack(np.stack(training_batch[:, 0])[..., 2]),
                                                                        [-1, 1])

                            action_train = training_batch[:, 1]

                            feed_dict = {mainQN.input_image_sequence: next_train_image_sequence_batch,
                                         mainQN.input_current_irradiance: next_train_curr_irr_batch,
                                         mainQN.input_current_control_input: next_train_control_input_batch,
                                         mainQN.targetQ: targetQ,
                                         mainQN.actions: action_train,
                                         mainQN.learning_rate: lr}

                            _, gradients, total_loss, q_values, lr = sess.run(
                                [mainQN.updateModel, mainQN.gradients, mainQN.loss, mainQN.Q, mainQN.learning_rate],
                                feed_dict=feed_dict)



                            ###################
                            # TARGET UPDATE
                            ###################
                            # Update targetnetwork by adding variables of training network to target variables, regularized by FLAGS.tau
                            Qnetwork.update_target_network(target_assign_operations, sess)

                            mean_max_q_value = np.mean( #mean over batch)
                                q_values)  # q_values is a batch of maximum action-values, take mean of it
                            episode_mean_q_value_list.append(mean_max_q_value)

                            if total_steps % FLAGS.summaries_every_n_total_steps == 0:
                                rl_logging.save_summaries(sess, merge_op, feed_dict, train_writer, gradients,
                                                          learning_rate=lr,
                                                          step=total_steps)

                    state = next_state  # the agent chooses actions based on the next state and the environment gives a next state

                    episode_reward_list.append(reward)  # reward when stepping through environment
                    chosen_action_list.append(action)
                    ###################
                    # LOGGING
                    ###################


                    if (episode_steps) % FLAGS.log_every_n_steps == 0:
                        # reward,#qvalue,#loss,time,speed per batch
                        curr_time = time.time()
                        duration = curr_time - start_time
                        sec_per_batch = float(duration / FLAGS.log_every_n_steps)
                        rl_logging.step_log(train_writer, epoch, (episode_nr + 1), num_episodes_per_epoch, total_steps,
                                            episode_steps, total_loss, reward, mean_max_q_value, sec_per_batch, lr, e)
                        start_time = time.time()

                # Episode done#########################################################


                if FLAGS.render_ep:
                    render_episode(env.current_train_control_inputs, env.current_train_episode)

                if total_steps > FLAGS.pre_train_steps:
                    if len(episode_reward_list) > 0 and len(
                            episode_mean_q_value_list) > 0:  # if checkpoint is loaded or pretraining phase, this lists will be empty
                        epoch_reward_list.append(sum(episode_reward_list) / len(episode_reward_list))
                        epoch_q_value_list.append(sum(episode_mean_q_value_list) / len(
                            episode_mean_q_value_list))  # based on the sampled batches! unlike in validation step

                    ###################
                    # LOGGING
                    ###################

                    rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=episode_reward_list,
                                               episodes_q_value_list=episode_mean_q_value_list, step=episode_nr,
                                               set="training_episode",
                                               action_counter=collections.Counter(chosen_action_list))

                    ###################
                    # SAVER
                    ###################

                    if (episode_nr + 1) % FLAGS.save_checkpoint_after_n_episodes == 0:
                        saver_episode_start_op = tf.assign(tf_episode_nr_start,
                                                           episode_nr + 1)  # next episode_nr on which a pretrained model will start.
                        saver_episode_start = sess.run(saver_episode_start_op)
                        print("Checkpoint at episode " + str(episode_nr + 1))
                        saver.save(sess, FLAGS.train_dir,
                                   global_step=saver_episode_start)  # start from next episode after loading checkpoint => +2

                    print("############ Finished episode number", episode_nr + 1)

                    ###################
                    # VALIDATION SESSION
                    ###################

                    if (episode_nr + 1) % FLAGS.validation_each_n_episodes == 0:
                        do_validation_run(train_writer=train_writer, sess=sess, network=mainQN, env=env,
                                          step=(episode_nr + 1))


#######################################################################################################################
if __name__ == '__main__':
    print("Start")
    tf.app.run()

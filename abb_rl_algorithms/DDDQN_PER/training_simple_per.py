import tensorflow as tf
import os, csv
import datetime as dt
from network import Qnetwork
from environment import Environment
from experience_replay_buffer import PrioritizedExperienceReplayBuffer
# from experience_replay_buffer import ExperienceReplayBuffer
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


tf.app.flags.DEFINE_string(
    'description', 'finetuine 2 , larger exploration and mpc ',
    'decribe experiment')


tf.app.flags.DEFINE_integer(
    'seed', 1338,
    'seed')

seed = FLAGS.seed
np.random.seed(seed)  # Reproducibility
random.seed(seed)



tf.app.flags.DEFINE_float(
    'per_process_gpu_memory_fraction', 0.05,
    'fraction of gpu memory used for this process')
# run_simple_irr_dqn_hl_mse_tau_0001_e_200_followirr_randomexp_sunspot_action_naive_nopt_mpc_divi_mask_uf1/
tf.app.flags.DEFINE_string(
    'train_dir',
    '/home/dladmin/Documents/arthurma/test_rl/finetune2/action_space_1_per/guided_normalized_input/mpc40_lr00025_tau10e3_div1000_clippedreward_per_hardsample_long_exp3/',
    'Directory where checkpoints, info, and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrained_checkpoint_path'
    , "/home/dladmin/Documents/arthurma/experiments_rl_final/action_space_1/guided_normalized_input/mpc30_lr00025_tau10e3_div1000/-9100"
    , 'The path to a checkpoint from which to fine-tune. Only if restore_latest_checkpoint is false')

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
# '/home/dladmin/Documents/arthurma/rf/low_reward_episodes200.pickle'
tf.app.flags.DEFINE_string(
    'load_train_episodes'
    ,'/home/dladmin/Documents/arthurma/rf/low_reward_episodes200.pickle',
    'Directory where File with test set is')

# "/home/dladmin/Documents/arthurma/rf/low_reward_episodes200.pickle"

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
    'batch_size', 32,
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

tf.app.flags.DEFINE_float(
    'divide_irr_ci', 1000.0,
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

tf.app.flags.DEFINE_boolean(
    'duelling', True,
    'Duelling or not')

################################
# Training
################################


tf.app.flags.DEFINE_float(
    'learning_rate', 0.00025,
    'Learning rate for gradient updates')

tf.app.flags.DEFINE_float(
    'huber_delta', 1.0,
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
    'pre_train_steps',1000,  # to small will lead to divergence because of non i i d samples
    'Number of random steps before training begins (Fill buffer)')

tf.app.flags.DEFINE_integer(
    'update_frequency', 1,
    'frequency of training steps while acting in the environment')
tf.app.flags.DEFINE_float(
    'discount_factor', 0.99,
    'Discount on target Q-values')

tf.app.flags.DEFINE_integer(
    'target_update_steps',None,
    'Set None to use Tau, otherwise will use fixed update steps of target network!')

tf.app.flags.DEFINE_float(
    'tau', 0.001,  # 0.00001 # too large will lead to divergence because target network follows main network too fast
    'target_update_steps needs to be None!, Convergence parameter, how much of main network is copied to target network')


tf.app.flags.DEFINE_integer(
    'replay_buffer_size',1000,
    'Max number of experiences in the experience replay buffer')

tf.app.flags.DEFINE_float(
    'l2_regularizer', 0.00004,
    'Regularization in training network')


tf.app.flags.DEFINE_float(
    'keep_prob', 1.0,
    'adam config')

tf.app.flags.DEFINE_bool(
    'train_value_only',False,
    'train value function only (for baseline and debugging)')


################################
# OPTIMIZER
################################

tf.app.flags.DEFINE_string(
    'optimizer', "adam",
    'adam,rmsprop,sgd')

tf.app.flags.DEFINE_float(
    'adam_epsilon', 0.01,
    'adam config')

tf.app.flags.DEFINE_float(
    'gradient_clipping', 0.0,
    'off if set to 0.0 => clip gradient -/+ value')

################################
# Exploration Strategy
################################
tf.app.flags.DEFINE_float(
    'start_e_greedy', 0.9,
    'epsilon greedy exploration strategy (start probability for exploration)')
tf.app.flags.DEFINE_float(
    'end_e_greedy',0.01,
    'epsilon greedy exploration strategy (start probability for exploration)')
tf.app.flags.DEFINE_integer(
    'annealing_steps', 10000000,
    'How many training steps to reduce start_e to end_e')

tf.app.flags.DEFINE_float(
    'validation_end_e_greedy', 0.1,
    'fixed exploration probability for validation set')

tf.app.flags.DEFINE_float(
    'mpc_guided_exploration',0.50,
    'probability with which action is selected that follows MPC during exploration')


tf.app.flags.DEFINE_string(
    'exploration_follow', 'MPC',
    'At beginning of each training episode. Start first controlinput at IRR or MPC')

tf.app.flags.DEFINE_integer(
    'start_exploration_deviation',50,
    'How much control input deviation (+/-) from exploration follow (chooses  uniform random between +/- value)')


################################
# Action Space/Rewards
################################
tf.app.flags.DEFINE_integer(
    'num_actions', 7,
    'Nr of actions to choose from')

# Enforces ramp restrictions: 100.0: 10% per minute => environment calculates depending on the time change between states which is roughly 6-8 seconds
tf.app.flags.DEFINE_float(
    'max_ramp_change_per_minute', 100.0,
    'Assuming 3 actions: Stay put, change battery throughput up by max_ramp_change or down by -max_ramp_change')

tf.app.flags.DEFINE_integer(
    'action_space', 1,
    '0: three action simple, 1: 0 follows irr, 7 actions, 2 0 follows irr only if irr is reachable otherwise straight, 7 actions')

tf.app.flags.DEFINE_integer(
    'reward_type', 1,
    '0: negative difference control/irr, 2: 1/abs(irr-input) positive between 0 and 1')


################################
# Logging and Checkpoints
################################

tf.app.flags.DEFINE_integer(
    'validation_each_n_episodes',  # 782/152, 2178 in hard set(100), 3640 for 200 (1401 hard = 38%), 7206 for 100
    30,
    'how often to go over validation set')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summaries_every_n_total_steps', 1500,
    'The frequency with which summaries are saved, in total_steps, None to switch off.')

tf.app.flags.DEFINE_integer(
    'save_checkpoint_after_n_episodes', 1401,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_nr_checkpoints_saved', 200,
    'Maximum number of newest checkpoints that are saved, older ones are deleted.')

tf.app.flags.DEFINE_boolean(
    'render_ep', False,
    'Render current train episode')

tf.app.flags.DEFINE_boolean(
    'trigger_file', True,
    'change learning rate or render episode on the fly..')


# nach gewisser anzahl episoden ...  / epochen  = nr. episodes ...


######################################################################################
##EPISODE RENDER
######################################################################################

def render_episode(current_control_input, current_episode):
    irr_list, mpc_list, ci_list, index_list = [], [], [], []
    # column_list = ["irr", "mpc", "ci"]

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
                         feed_dict={network.input_image_sequence: (np.array([state[0]]))*(1/FLAGS.divide_image_values),
                                    network.input_current_irradiance: np.reshape(
                                        [state[1]],
                                        [-1, FLAGS.img_sequence_length]) *(1/FLAGS.divide_irr_ci),
                                    network.input_current_control_input: np.reshape(
                                        [state[2]],
                                        [-1, 1])*(1/FLAGS.divide_irr_ci),network.keep_prob:1.0})

            chosen_action_list.append(action[0])

            print("Val action:",action)

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
                               episodes_mean_max_q_value_list=episodes_q_value_list, episodes_mean_chosen_q_value_list=None, episodes_mean_batch_reward_list=None, step=step, action_counter=action_counter,
                               set="validation_epoch")


#######################################################################################################################
##HELPERS
#######################################################################################################################

def get_train_feed_dict(network, batch, targetQ, learning_rate):
    # state_idx = 0 => current, state_idx = 3 => next
    next_train_image_sequence_batch = np.stack(np.stack(batch[:, 0])[..., 0])*(1/FLAGS.divide_image_values)
    next_train_curr_irr_batch = np.stack(np.stack(batch[:, 0])[..., 1])*(1/FLAGS.divide_irr_ci)
    next_train_control_input_batch = np.reshape(np.stack(np.stack(batch[:, 0])[..., 2]), [-1, 1])*(1/FLAGS.divide_irr_ci)
    action_train = batch[:, 1]

    feed_dict = {network.input_image_sequence: next_train_image_sequence_batch,
                 network.input_current_irradiance: next_train_curr_irr_batch,
                 network.input_current_control_input: next_train_control_input_batch,
                 network.targetQ: targetQ,
                 network.actions: action_train,
                 network.learning_rate: learning_rate,
                 network.keep_prob: FLAGS.keep_prob}

    return feed_dict


def get_target_q(sess, mainQN, targetQN, training_batch):
    next_image_sequence_batch = np.stack(np.stack(training_batch[:, 3])[..., 0])*(1/FLAGS.divide_image_values)
    # misc.imshow(next_image_sequence_batch[6][:,:,0:3])
    next_curr_irr_batch = np.stack(np.stack(training_batch[:, 3])[..., 1])*(1/FLAGS.divide_irr_ci)
    next_control_input_batch = np.reshape(np.stack(np.stack(training_batch[:, 3])[..., 2]), [-1, 1])*(1/FLAGS.divide_irr_ci)

    feed_dict_mqn = {mainQN.input_image_sequence: next_image_sequence_batch,
                     mainQN.input_current_irradiance: next_curr_irr_batch,
                     mainQN.input_current_control_input: next_control_input_batch,
                     mainQN.keep_prob: 1.0}

    feed_dict_tqn = {targetQN.input_image_sequence: next_image_sequence_batch,
                     targetQN.input_current_irradiance: next_curr_irr_batch,
                     targetQN.input_current_control_input: next_control_input_batch,
                     mainQN.keep_prob: 1.0}

    ### Calculate the target network value y (next state,next action maximum) (rhs of loss function)
    Q1 = sess.run(mainQN.predict, feed_dict=feed_dict_mqn)  # select best action indices
    Q2 = sess.run(targetQN.Qout,
                  feed_dict=feed_dict_tqn)  # calculate all Q values of all action outputs in target network

    end_multiplier = -(training_batch[:, 4] - 1)  # 1 if d = false  0 if d = True
    doubleQ = Q2[range(  # Double Q Learning
        training_batch.shape[
            0]), Q1]  # select the Q values of target network according to best action arguments of training network
    # doubleQ => batchsize x 1  with Q values of best actions according to training network


    targetQ = training_batch[:, 2] + (
        FLAGS.discount_factor * doubleQ * end_multiplier)  # y = reward + disc_f*max(Q(s',a',theta))

    # Calculate Q(s,a,theta) and then get loss with targetQ as input, train on loss



    return targetQ


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

    # create trigger file
    t_data = [FLAGS.learning_rate, FLAGS.render_ep]
    t_cols = ["lr", "re"]

    trigger_pd = pd.DataFrame(columns=t_cols, data=[t_data])

    trigger_pd.to_csv(os.path.join(FLAGS.train_dir, "trigger_file.csv"))

    ###################
    # Initialize
    ###################

    with tf.device('/gpu:0'):

        config = tf.ConfigProto(
            device_count={'GPU': 1},
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        )

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # Environment and Networks
        env = Environment(train_set_path=FLAGS.train_set_path, test_set_path=FLAGS.test_set_path,
                          episode_length_train=FLAGS.episode_length_train,
                          episode_length_test=FLAGS.episode_length_test,
                          sequence_length=FLAGS.img_sequence_length, sequence_stride=FLAGS.img_sequence_stride,
                          actions=FLAGS.num_actions,
                          image_size=FLAGS.img_size, action_space=FLAGS.action_space, file=FLAGS.data_file,
                          load_train_episodes=FLAGS.load_train_episodes, load_test_episodes=FLAGS.load_test_episodes,
                          mask_path=FLAGS.mask,
                          divide_image_values=FLAGS.divide_image_values,
                          sample_training_episodes=FLAGS.sample_train_episodes,
                          exploration_follow=FLAGS.exploration_follow, start_exploration_deviation=FLAGS.start_exploration_deviation,
                          reward_type = FLAGS.reward_type)

        mainQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size,
                          img_size=FLAGS.img_size,
                          img_sequence_len=FLAGS.img_sequence_length, huber_delta=FLAGS.huber_delta,
                          adam_epsilon=FLAGS.adam_epsilon, add_irr=FLAGS.add_irr, train_value_only=FLAGS.train_value_only,duelling=FLAGS.duelling,gradient_clipping=FLAGS.gradient_clipping,optimizer=FLAGS.optimizer)

        targetQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size,
                            img_size=FLAGS.img_size,
                            img_sequence_len=FLAGS.img_sequence_length, huber_delta=FLAGS.huber_delta,
                            adam_epsilon=FLAGS.adam_epsilon, add_irr=FLAGS.add_irr, train_value_only=FLAGS.train_value_only,duelling=FLAGS.duelling,gradient_clipping=FLAGS.gradient_clipping,optimizer=FLAGS.optimizer)



        if FLAGS.network == "simple_duelling_dqn":
            mainQN.simple_duelling_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_main')
            targetQN.simple_duelling_dqn(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_target')

        elif FLAGS.network == "simple_duelling_dqn_old":
            mainQN.simple_duelling_dqn_old(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_main')
            targetQN.simple_duelling_dqn_old(regularizer=FLAGS.l2_regularizer, scope='simple_duelling_dqn_target')
        else:
            raise ValueError("Illegal architecture")

        trainables = tf.trainable_variables()

        target_assign_operations = Qnetwork.target_update_operations(trainable_variables=trainables, tau=FLAGS.tau)
        target_full_assign_operations = Qnetwork.target_full_update_operations(trainable_variables=trainables)

        # Buffer to sample experience batches from (off-policy learning required)
        global_exp_buffer = PrioritizedExperienceReplayBuffer(buffer_size=FLAGS.replay_buffer_size)

        # Initialize e_greedy exploration probability
        e = FLAGS.start_e_greedy
        e_reduction = (FLAGS.start_e_greedy - FLAGS.end_e_greedy) / FLAGS.annealing_steps

        # List of rewards and steps per episode
        epoch_reward_list = list()
        epoch_mean_max_q_value_list = list()
        epoch_mean_batch_reward_list = list()
        epoch_mean_chosen_q_value_list = list()
        total_steps = 0
        num_episodes = env.nr_train_episodes * FLAGS.num_epochs  # how many episodes to run, each episode is visited before an episode is visited twice...
        num_episodes_per_epoch = env.nr_train_episodes

        tf_episode_nr_start = tf.get_variable("episode_counter", initializer=0)

        lr = FLAGS.learning_rate

        render_ep_t = False

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

        epn = tf.placeholder(dtype=tf.int32, shape=None)
        saver_episode_start_op = tf.assign(tf_episode_nr_start, epn)

        init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:
            sess.run(init)
            sess.graph.finalize()
            Qnetwork.update_target_network(target_full_assign_operations,
                                           sess=sess)  # initialize target network, setting it equal to main network

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
                    if len(epoch_reward_list) > 0 and len(epoch_mean_max_q_value_list) > 0 and len(epoch_mean_batch_reward_list)>0 and len(epoch_mean_chosen_q_value_list)>0:
                        print("Epoch finished: save training statistics")
                        rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=epoch_reward_list,
                                                   episodes_mean_max_q_value_list=epoch_mean_max_q_value_list,episodes_mean_chosen_q_value_list=epoch_mean_chosen_q_value_list, episodes_mean_batch_reward_list=epoch_mean_batch_reward_list, step=epoch,
                                                   set="training_epoch")
                    epoch_reward_list, epoch_mean_max_q_value_list,epoch_mean_batch_reward_list,epoch_mean_chosen_q_value_list = list(), list(),list(),list()

                print("Epoch: ", str(epoch), "/", str(FLAGS.num_epochs))
                print("Episode number: ", str(episode_nr + 1))

                done = 0
                episode_reward_list = list()
                chosen_action_list = list()
                episode_mean_max_q_value_list = list()
                episode_mean_chosen_q_value_list = list()
                episode_mean_batch_reward_list = list()
                episode_mean_action_q_value_list = list()
                episode_steps = 0
                mean_max_q_value = 0
                mean_chosen_q_value = 0
                total_loss = 0
                state, abort = env.reset()

                if FLAGS.trigger_file:
                    trigger_path = os.path.join(FLAGS.train_dir, 'trigger_file.csv')
                    trigger_info = pd.read_csv(trigger_path)

                    lr = trigger_info['lr'].values[0]
                    render_ep_t = trigger_info['re'].values[0]

                    print(trigger_info, render_ep_t)

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

                        print("MPC action:", action)
                    else:
                        # Predict Action
                        action = \
                        sess.run(mainQN.predict, feed_dict={mainQN.input_image_sequence: (np.array([state[0]]))*(1/FLAGS.divide_image_values),
                                                            mainQN.input_current_irradiance: np.reshape(state[1],
                                                                                                        [-1,
                                                                                                         FLAGS.img_sequence_length])*(1/FLAGS.divide_irr_ci),
                                                            mainQN.input_current_control_input: np.reshape(
                                                                state[2], [-1, 1])*(1/FLAGS.divide_irr_ci),mainQN.keep_prob:1.0})[0]
                        print("Network action:", action)

                    next_state, reward, done = env.step(action)  # integer of action buffersize 50000

                    experience = np.reshape(np.array([state, action, reward, next_state, done]), [1, 5])

                    # Calculate td_loss for PER
                    targetQ = get_target_q(sess=sess, mainQN=mainQN, targetQN=targetQN, training_batch=experience)
                    feed_dict = get_train_feed_dict(network=mainQN, batch=experience, targetQ=targetQ, learning_rate=lr)
                    time_diff_error = np.abs(sess.run(mainQN.td_error, feed_dict=feed_dict))

                    global_exp_buffer.add(time_diff_error, experience)

                    if total_steps > FLAGS.pre_train_steps:  # create enough samples in experience replay buffer before training
                        if e > FLAGS.end_e_greedy:  # if end epsilon not reached, reduce e
                            e -= e_reduction

                        if total_steps % (FLAGS.update_frequency) == 0:
                            training_batch_full = global_exp_buffer.sample(FLAGS.batch_size, 5)

                            training_batch = np.vstack(training_batch_full[:, 2])

                            # print(training_batch)
                            targetQ = get_target_q(sess=sess, mainQN=mainQN, targetQN=targetQN,
                                                   training_batch=training_batch)

                            feed_dict = get_train_feed_dict(network=mainQN, batch=training_batch, targetQ=targetQ,
                                                            learning_rate=lr)

                            _, gradients, time_diff_error, total_loss, q_values, lr,qout_val = sess.run(
                                [mainQN.updateModel, mainQN.gradients, mainQN.td_error, mainQN.loss, mainQN.Q,
                                 mainQN.learning_rate,mainQN.Qout],
                                feed_dict=feed_dict)

                            # Update priorities in sum-tree of PER
                            indices = training_batch_full[:, 0]

                            for idx, error in zip(indices, time_diff_error):
                                global_exp_buffer.update(idx, np.abs(error))

                            # Update targetnetwork by adding variables of training network to target variables, regularized by FLAGS.tau



                            if FLAGS.target_update_steps is None:
                                #USE TAU to gradually update target network if no update step is set
                                Qnetwork.update_target_network(target_assign_operations, sess)

                            else:
                                if total_steps % FLAGS.target_update_steps == 0:
                                    #copy whole main network to target
                                    Qnetwork.update_target_network(target_full_assign_operations,sess=sess)


                            mean_chosen_q_value = np.mean(  # mean over batch)
                                q_values)  # q_values is a batch of maximum action-values, take mean of it

                            mean_max_q_value = np.mean(np.max(qout_val,axis=1))

                            mean_action_q_value = np.mean(qout_val,axis=0)

                            print("Mean action Q-values:",mean_action_q_value)


                            episode_mean_max_q_value_list.append(mean_max_q_value)
                            episode_mean_batch_reward_list.append(np.mean(training_batch[:, 2]))
                            episode_mean_chosen_q_value_list.append(mean_chosen_q_value)
                            episode_mean_action_q_value_list.append(mean_action_q_value)


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
                                            episode_steps, total_loss, reward, mean_max_q_value,mean_chosen_q_value, sec_per_batch, lr, e)
                        start_time = time.time()

                # Episode done#########################################################


                if FLAGS.render_ep or render_ep_t:
                    render_episode(env.current_train_control_inputs, env.current_train_episode)

                if total_steps > FLAGS.pre_train_steps:
                    if len(episode_reward_list) > 0 and len(
                            episode_mean_max_q_value_list) > 0 and len(episode_mean_batch_reward_list)>0 and len(episode_mean_chosen_q_value_list)>0:  # if checkpoint is loaded or pretraining phase, this lists will be empty
                        epoch_reward_list.append(sum(episode_reward_list) / len(episode_reward_list))
                        epoch_mean_max_q_value_list.append(sum(episode_mean_max_q_value_list) / len(
                            episode_mean_max_q_value_list))  # based on the sampled batches! unlike in validation step
                        epoch_mean_batch_reward_list.append(sum(episode_mean_batch_reward_list)/len(episode_mean_batch_reward_list))
                        epoch_mean_chosen_q_value_list.append(sum(episode_mean_chosen_q_value_list)/(len(episode_mean_chosen_q_value_list)))

                    ###################
                    # LOGGING
                    ###################

                    rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=episode_reward_list,
                                               episodes_mean_max_q_value_list=episode_mean_max_q_value_list,episodes_mean_chosen_q_value_list=episode_mean_chosen_q_value_list,episodes_mean_batch_reward_list=episode_mean_batch_reward_list,episode_mean_action_q_value_list=episode_mean_action_q_value_list, step=episode_nr,
                                               set="training_episode",
                                               action_counter=collections.Counter(chosen_action_list))

                    ###################
                    # SAVER
                    ###################

                    if (episode_nr + 1) % FLAGS.save_checkpoint_after_n_episodes == 0:
                        # next episode_nr on which a pretrained model will start.
                        saver_episode_start = sess.run(saver_episode_start_op, feed_dict={epn: episode_nr + 1})
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

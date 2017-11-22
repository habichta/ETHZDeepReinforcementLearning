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
slim = tf.contrib.slim
from scipy import misc
import pandas as pd



FLAGS = tf.app.flags.FLAGS

################################
# General
################################

tf.app.flags.DEFINE_float(
    'per_process_gpu_memory_fraction', 0.3,
    'fraction of gpu memory used for this process')

tf.app.flags.DEFINE_string(
    'eval_dir', '/home/nox/Drive/Documents/Masterarbeit/rf/runs/default6_eval/',
    'Directory where checkpoints, info, and event logs are written to.')

tf.app.flags.DEFINE_string(
    'pretrained_checkpoint_path',
"/media/nox/OS/Linux/Documents/Masterarbeit/rf/runs/default6/-103",
    'The path to a checkpoint from which to fine-tune. Only if restore_latest_checkpoint is false')

tf.app.flags.DEFINE_string(
    'test_set_path',
    '/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/abb_rl_algorithms/DDDQN/validation_list.out',
    'Directory where File with test set is')

tf.app.flags.DEFINE_string(
    'train_set_path',
    '/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/abb_rl_algorithms/DDDQN/train_list.out',
    'Directory where File with train set is')

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

################################
# Simple QNetwork
################################
tf.app.flags.DEFINE_float(
    'stream_hidden_layer_size', 256,
    'hidden layer size before Q-value regression')

################################
# Action Space
################################
tf.app.flags.DEFINE_integer(
    'num_actions', 3,
    'Nr of actions to choose from')  # keep 3!

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
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summaries_every_n_total_steps', 1000,
    'The frequency with which summaries are saved, in total_steps, None to switch off.')


######################################################################################
##TESTING
######################################################################################

def do_validation_run(sess, network, env, train_writer, step, output_path):
    nr_validation_episodes = env.nr_test_episodes
    print("Validation run on " + str(nr_validation_episodes) + " episodes...")

    total_episodes_reward_list = list()
    total_episodes_q_value_list = list()
    total_chosen_action_list = list()
    control_input_df_list = list()

    for episode_nr in range(nr_validation_episodes):
        print("Validation Episode " + str(episode_nr + 1) + "/" + str(nr_validation_episodes))

        state, abort = env.test_reset()  # get next episode, initialize, TEST SET
        done = 0

        episode_reward_sum = 0
        episode_q_value_sum = 0
        episode_steps = 0
        episode_chosen_action_list = list()
        episode_reward_list = list()
        episode_q_value_list = list()

        while done == 0 and not abort:
            episode_steps += 1

            action, action_value_qs = \
                sess.run([network.predict, network.Qout], feed_dict={network.input_image_sequence: np.array([state[0]]),
                                                                     network.input_current_irradiance: np.reshape(
                                                                         [state[1]],
                                                                         [-1, FLAGS.img_sequence_length]),
                                                                     network.input_current_control_input: np.reshape(
                                                                         [state[2]],
                                                                         [-1, 1])})

            # TODO: extract V , Advantage  of current state as well.. maybe also directly TSNE embedding on some of the inputs.. save coordinates in csv

            mean_max_action_value_q = np.mean(np.max(action_value_qs, axis=1))

            next_state, reward, done = env.test_step(action) #step TEST SET

            episode_reward_sum += reward
            episode_q_value_sum += mean_max_action_value_q

            episode_reward_list.append(reward)
            episode_q_value_list.append(mean_max_action_value_q)
            episode_chosen_action_list.append(action[0])

            total_chosen_action_list.append(action[0])

            state = next_state

        print("Cumulative episode " + str(episode_nr + 1) + " reward:", episode_reward_sum)
        print("Average episode " + str(episode_nr + 1) + " reward:", episode_reward_sum / episode_steps)
        print("Average episode " + str(episode_nr + 1) + " Q Value:", episode_q_value_sum / episode_steps)
        total_episodes_reward_list.append(episode_reward_sum / episode_steps)
        total_episodes_q_value_list.append(episode_q_value_sum / episode_steps)

        episode_action_counter = collections.Counter(total_chosen_action_list)
        rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=episode_reward_list,
                                   episodes_q_value_list=episode_q_value_list, step=episode_nr,
                                   action_counter=episode_action_counter,
                                   set="validation_episode")

        # Calculate control input dataframe, interpolate linearly to seconds

        control_inputs_tuple = network.env.current_episode_test_control_input_values

        c_inputs = [t[0] for t in control_inputs_tuple]
        c_index = [t[1] for t in control_inputs_tuple]

        ci_df = pd.DataFrame(data=c_inputs, index=c_index, columns=["ci"])
        ci_df = ci_df.asfreq('S')
        ci_df = ci_df.astype(
            float).interpolate(method='time')


        control_input_df_list.append(ci_df)

    action_counter = collections.Counter(total_chosen_action_list)

    total_control_input_df = pd.concat(control_input_df_list, axis=0).sort_index()

    total_control_input_df.to_csv(os.path.join(output_path, "eval_predictions.csv"))

    rl_logging.save_statistics(train_writer=train_writer, episodes_reward_list=total_episodes_reward_list,
                               episodes_q_value_list=total_episodes_q_value_list, step=step,
                               action_counter=action_counter,
                               set="validation_epoch", write_path=output_path)


#######################################################################################################################
##MAIN
#######################################################################################################################



def main(_):
    if tf.gfile.Exists(FLAGS.eval_dir):
        raise ValueError("Eval directory exists already")

    os.makedirs(FLAGS.eval_dir)

    with open(os.path.join(FLAGS.eval_dir, "test_model_info.csv"), "w+") as f:
        w = csv.writer(f)
        w.writerow([str(dt.datetime.now())])
        for key, val in FLAGS.__flags.items():
            w.writerow([key, val])

    ###################
    # Initialize
    ###################

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.reset_default_graph()

    # Environment and Networks
    env = Environment(train_set_path=FLAGS.train_set_path, test_set_path=FLAGS.test_set_path,
                      episode_length=None, sequence_length=FLAGS.img_sequence_length,
                      image_size=FLAGS.img_size,follow_irr_actions=FLAGS.follow_irr_actions)

    mainQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size, img_size=FLAGS.img_size,
                      img_sequence_len=FLAGS.img_sequence_length)
    mainQN.simple_duelling_dqn(regularizer=0.0, scope='simple_duelling_dqn_main')

    targetQN = Qnetwork(environment=env, stream_hidden_layer_size=FLAGS.stream_hidden_layer_size,
                        img_size=FLAGS.img_size,
                        img_sequence_len=FLAGS.img_sequence_length)
    targetQN.simple_duelling_dqn(regularizer=0.0, scope='simple_duelling_dqn_target')

    ###################
    # TESTING SESSION
    ###################
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('Loading pretrained model...')
        saver.restore(sess, FLAGS.pretrained_checkpoint_path)

        # Tensorboard summary writer
        train_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                             sess.graph)

        do_validation_run(train_writer=train_writer, sess=sess, network=mainQN, env=env, step=1,
                          output_path=FLAGS.eval_dir)


#######################################################################################################################
if __name__ == '__main__':
    print("Start")
    tf.app.run()

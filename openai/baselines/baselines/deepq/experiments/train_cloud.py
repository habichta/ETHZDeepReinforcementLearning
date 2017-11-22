import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from cloud_environment import  CloudEnvironment
import numpy as np
import collections
import os
import csv
import random


#Logging


def logger_callback(locals,globals):

    training_started = locals['log_training_started']
    trained = locals['log_trained']
    done = locals['done']
    num_episodes = locals['num_episodes']



    if training_started:

        log_action_l = locals['log_action_l'] # actions chosen in current episode step
        log_action_l.append(locals['action'])

        log_errors_l = locals['log_errors_l']  # Huber loss of td errors
        log_q_t_selected_l = locals['log_q_t_selected_l']
        log_q_t_targets_l = locals['log_q_t_targets_l_l']
        log_td_errors_l = locals['log_td_errors_l']  # difference between state 1 and  next state predictions
        log_errors_l = locals['log_errors_l']  # Huber loss of td errors

        if trained:
             #selected actions in batch
            log_q_t_selected_l.append(np.mean(locals['log_q_t_selected']))
            log_q_t_targets_l.append(np.mean(locals['log_q_t_targets']))
            log_td_errors_l.append(np.mean(locals['log_td_errors']))
            log_errors_l.append(np.mean(locals['log_errors']))


        if done:
            rew500 = locals['mean_500ep_reward']
            rew100 = locals['mean_100ep_reward']
            rew50 = locals['mean_50ep_reward']
            rew10 = locals['mean_10ep_reward']

            logger = globals['logger']

            action_counter = collections.Counter(log_action_l)

            episode_q_t_selected = np.mean(log_q_t_selected_l) #1x1
            episode_q_t_targets = np.mean(log_q_t_targets_l) #1x1
            #episode_q_t = np.mean(log_q_t_l,axis=0) #1x2

            episode_td_errors = np.mean(log_td_errors_l) #1x1
            episode_errors = np.mean(log_errors_l) # 1x1

            logger.record_tabular("Tmp File", locals['td'])
            logger.record_tabular("Actions Count", action_counter)
            logger.record_tabular("Mean selected Q",episode_q_t_selected)
            logger.record_tabular("Mean selected target Q",episode_q_t_targets)
            #logger.record_tabular("Mean Action Q", episode_q_t)
            logger.record_tabular("Mean TD Error", episode_td_errors)
            logger.record_tabular("Mean Huber Error", episode_errors)
            logger.dump_tabular()


            if num_episodes % 10 == 0:
                #Write log


                path = locals['train_file_path']

                print("Writing episode {} log to ".format(num_episodes), path)

                with open(path, 'a') as f:



                    env = locals['env']

                    ep_id = env.episode_id

                    actions_np = np.zeros(env.action_space.n)

                    for k,v in action_counter.items():
                        actions_np[k] = v


                    action_count_header = ['action_count{}'.format(i) for i in range(env.action_space.n)]
                    #action_q_header = ['mean_action_q{}'.format(i) for i in range(len(episode_q_t.tolist()))]

                    headers = ['episode_id','episode','steps','reward500','reward100','reward50','reward10','mean_s_q','mean_t_q','mean_td_error','mean_h_error']
                    #headers = headers + action_q_header+action_count_header
                    headers = headers  + action_count_header

                    steps = locals['t']

                    action_counts = list(actions_np)
                    #actions_qs = [q for q in episode_q_t.tolist()]

                    #output_list = [num_episodes]+[steps]+[rew100]+[rew50]+[rew10]+[episode_q_t_selected]+[episode_q_t_targets]+[episode_td_errors]+[episode_errors]+ actions_qs+action_counts
                    output_list = [ep_id]+ [num_episodes] + [steps] +[rew500]+ [rew100] + [rew50] + [rew10] + [episode_q_t_selected] + [
                        episode_q_t_targets] + [episode_td_errors] + [episode_errors] + action_counts
                    print(headers)
                    print(output_list)
                    w = csv.writer(f)

                    if os.stat(path).st_size == 0:
                        w.writerow(headers)

                    w.writerow(output_list)

    return False



def main():
    np.random.seed(1)
    random.seed(1)

    channels=3
    seq_length=2
    img_size=84

    env = CloudEnvironment(img_size=img_size,radius=[10,20],sequence_stride=1,channels=channels,sequence_length=seq_length,ramp_step=0.1,action_type=0,action_nr=2,stochastic_irradiance=True,save_images=False)
    #Note: cloud speed can be changes but may also require different ramps.. default, speed of cloud per frame at least 1 pixel in  y direction
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        channels=channels,
        seq_length=seq_length,
        img_size=img_size
    )
    act = deepq.learn(
        train_file_path='train_log.csv',
        env=env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000,
        buffer_size=101,
        initial_exploration=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=100,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=logger_callback,
        load_cpk=None
        #"/media/nox/OS/Linux/Documents/Masterarbeit/simple_rl/model1/model"
    )
    act.save("cloud_model.pkl")


if __name__ == '__main__':
    main()

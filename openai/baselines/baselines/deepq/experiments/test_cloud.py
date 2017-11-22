import gym

from baselines import deepq

from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from cloud_environment import  CloudEnvironment
import numpy as np
import collections
import os
import csv
import pandas as pd


#Logging


def logger_callback(locals,globals):


    done = locals['done']
    num_episodes = locals['num_episodes']



    log_action_l = locals['log_action_l'] # actions chosen in current episode step
    log_action_l.append(locals['action'])


    if done:

            action_counter = collections.Counter(log_action_l).items()

            reward_sum = np.sum(locals['episode_rewards'])
            reward_mean = np.mean(locals['episode_rewards'])
            c_reward_sum = np.sum(locals['cumulative_episode_rewards'])
            c_reward_mean = np.mean(locals['cumulative_episode_rewards'])

            path = locals['test_file_path']

            print("Writing episode {} log to ".format(num_episodes), path)

            with open(path, 'a') as f:

                env = locals['env']

                actions_np = np.zeros(env.action_space.n)

                for k, v in action_counter:
                    actions_np[k] = v

                action_count_header = ['action_count{}'.format(i) for i in range(env.action_space.n)]

                #action_q_header = ['mean_action_q{}'.format(i) for i in range(len(episode_q_t.tolist()))]

                headers = ['episode','reward_sum','reward_mean','c_reward_sum','c_reward_mean']
                #headers = headers + action_q_header+action_count_header
                headers = headers  + action_count_header


                action_counts = list(actions_np)
                #actions_qs = [q for q in episode_q_t.tolist()]

                #output_list = [num_episodes]+[steps]+[rew100]+[rew50]+[rew10]+[episode_q_t_selected]+[episode_q_t_targets]+[episode_td_errors]+[episode_errors]+ actions_qs+action_counts
                output_list = [num_episodes] + [reward_sum] + [reward_mean] + [c_reward_sum] + [c_reward_mean]  + action_counts
                print(headers)
                print(output_list)
                w = csv.writer(f)

                if os.stat(path).st_size == 0:
                    w.writerow(headers)

                w.writerow(output_list)

    return False

def result_callback(ci_list,episode_list,locals,globals):

    nps = len(episode_list[0])-len(ci_list[0])

    cis_l = [[np.nan]*nps + cil for cil in ci_list]

    e_df = pd.concat(episode_list,axis=0).reset_index(drop=True)

    ci_df = pd.DataFrame(np.concatenate(cis_l),columns=['ci']).reset_index(drop=True)

    output_df = pd.concat([e_df,ci_df],axis=1)
    output_df.dropna(inplace=True)
    output_df = output_df.reset_index(drop=True)
    output_df.to_csv('eval_predictions.csv')



def main():
    load_cpk="/home/nox/Masterarbeit/thesis_data/baseline_rl/simple_rl/7_unbalanced_test/experiments_unbalanced/cloud_model.pkl"
    channels=3
    seq_length=2
    img_size=84

    env = CloudEnvironment(img_size=img_size,radius=[12,13],sequence_stride=1,channels=channels,sequence_length=seq_length,ramp_step=0.1,action_type=1,action_nr=3,stochastic_irradiance=True,save_images=True)
    #Note: cloud speed can be changes but may also require different ramps.. default, speed of cloud per frame at least 1 pixel in  y direction
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        channels=channels,
        seq_length=seq_length,
        img_size=img_size
    )
    deepq.test(load_cpk=load_cpk,
        result_callback=result_callback,
        env=env,
        q_func=model,
        log_callback=logger_callback,
        episode_n=1
    )

if __name__ == '__main__':
    main()

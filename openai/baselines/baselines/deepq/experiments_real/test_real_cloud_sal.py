import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from cloud_environment_real import RealCloudEnvironment
import numpy as np
import collections
import os
import csv
import pandas as pd
import dill as pickle


#Logging


def logger_callback(locals,globals):


    done = locals['done']
    num_episodes = locals['num_episodes']



    log_action_l = locals['log_action_l'] # actions chosen in current episode step
    log_action_l.append(locals['action'])


    if done:

            action_counter = collections.Counter(log_action_l)

            reward_sum = np.sum(locals['episode_rewards'])
            reward_mean = np.mean(locals['episode_rewards'])
            c_reward_sum = np.sum(locals['cumulative_episode_rewards'])
            c_reward_mean = np.mean(locals['cumulative_episode_rewards'])

            path = locals['test_file_path']

            print("Writing episode {} log to ".format(num_episodes), path)


            env = locals['env']

            if reward_sum <= -60.0:
                locals['ep_600'].append(env.current_train_episode)
            if reward_sum <= -50.0:
                locals['ep_500'].append(env.current_train_episode)
            if reward_sum <= -40.0:
                locals['ep_400'].append(env.current_train_episode)
            if reward_sum <= -30.0:
                locals['ep_300'].append(env.current_train_episode)
            if reward_sum <= -20.0:
                locals['ep_200'].append(env.current_train_episode)
            if reward_sum <= -10.0:
                locals['ep_100'].append(env.current_train_episode)
            if reward_sum <= -1.7:
                locals['ep_17'].append(env.current_train_episode)

            with open(path, 'a') as f:




                ep_id = env.episode_id
                ep_end_id = env.episode_end_id

                actions_np = np.zeros(env.action_space.n)

                for k, v in action_counter.items():
                    actions_np[k] = v

                action_count_header = ['action_count{}'.format(i) for i in range(env.action_space.n)]
                #action_q_header = ['mean_action_q{}'.format(i) for i in range(len(episode_q_t.tolist()))]

                headers = ['episode_id','episode_end_id','episode_nr','reward_sum','reward_mean','c_reward_sum','c_reward_mean']
                #headers = headers + action_q_header+action_count_header
                headers = headers  + action_count_header


                action_counts = list(actions_np)
                #actions_qs = [q for q in episode_q_t.tolist()]

                #output_list = [num_episodes]+[steps]+[rew100]+[rew50]+[rew10]+[episode_q_t_selected]+[episode_q_t_targets]+[episode_td_errors]+[episode_errors]+ actions_qs+action_counts
                output_list = [ep_id]+[ep_end_id]+[num_episodes] + [reward_sum] + [reward_mean] + [c_reward_sum] + [c_reward_mean]  + action_counts
                print(headers)
                print(output_list)
                w = csv.writer(f)

                if os.stat(path).st_size == 0:
                    w.writerow(headers)

                w.writerow(output_list)

    return False



def result_callback(ci_list,episode_list,locals,globals):

    cie_df_l = []

    for ci_e, e in zip(ci_list,episode_list):
        c_inputs = [t[0] for t in ci_e]
        c_index = [t[1] for t in ci_e]  # timestamps
        ci_df = pd.DataFrame(data=c_inputs, index=c_index, columns=["ci"])
        print(ci_df,ci_df.shape)
        e_df = e.loc[c_index]

        print(e_df,e_df.shape)

        cie_df = pd.concat([ci_df,e_df],axis=1)
        print(cie_df.shape)
        cie_df_l.append(cie_df)


    total_control_input_df = pd.concat(cie_df_l ,axis=0).sort_index()
    print(total_control_input_df.shape)
    total_control_input_df.to_csv("eval_predictions.csv")

    """
    with open('ep17_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_17'],f)

    with open('ep100_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_100'],f)

    with open('ep200_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_200'],f)

    with open('ep300_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_300'],f)

    with open('ep400_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_400'],f)

    with open('ep500_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_500'],f)

    with open('ep600_200_test.pkl','wb') as f:
        pickle.dump(locals['ep_600'],f)
    """










def main():

    load_cpk = 'cloud_model.pkl'

    data_path="/media/nox/OS/Linux/Documents/Masterarbeit/data/Daten/data_C_int/"
    mask_path="/media/nox/OS/Linux/Documents/Masterarbeit/data/Daten/img_C/cavriglia_skymask256.png"
    img_path="/media/nox/OS/Linux/Documents/Masterarbeit/data/Daten/img_C/"
    test_set_path ="test_list.out"

    channels=3
    seq_length=3
    img_size=84
    seq_stride=9

    env =   RealCloudEnvironment(data_path,img_path,test_set_path, image_size=img_size,
                 sequence_length=seq_length, sequence_stride=seq_stride, action_nr=3, action_type=1, ramp_step=0.1, episode_length_train=200,
                file="rl_data_sp.csv",load_train_episodes='ep600_200.pkl',mask_path=mask_path,exploration_follow="IRR",start_exploration_deviation=0.0,clip_irradiance=False)

    model = deepq.models_sal.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        channels=channels,
        seq_length=seq_length,
        img_size=img_size
    )
    deepq.test_sal(load_cpk=load_cpk,
        result_callback=result_callback,
        env=env,
        q_func=model,
        log_callback=logger_callback,
        episode_n = env.episode_n
    )


if __name__ == '__main__':
    main()

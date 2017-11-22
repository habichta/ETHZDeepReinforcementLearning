import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from cloud_environment_real import RealCloudEnvironment
import numpy as np
import collections
import os
import csv
import random

#NUR EINE EPISODE
#Bilder normalisieren? Brightness change jan  => HSV
#komplzierteres toy environment  transparenz
#hard data set wenn auf eines funktionert


#Logging


def logger_callback(locals,globals):

    training_started = locals['log_training_started']
    done = locals['done']
    num_episodes = locals['num_episodes']



    if training_started:

        log_action_l = locals['log_action_l'] # actions chosen in current episode step
        log_action_l.append(locals['action'])

        log_q_t_selected_l = locals['log_q_t_selected_l'] #selected actions in batch
        log_q_t_selected_l.append(np.mean(locals['log_q_t_selected']))

        log_q_t_targets_l = locals['log_q_t_targets_l_l'] #q values of actions selected by target network and double Q
        log_q_t_targets_l.append(np.mean(locals['log_q_t_targets']))

        """
        log_q_t_l = locals['log_q_t_l'] # q values of all actions in a batch
        log_q_t_l.append(np.mean(locals['log_q_t'],axis=0))
        """
        log_td_errors_l = locals['log_td_errors_l']  # difference between state 1 and  next state predictions
        log_td_errors_l.append(np.mean(locals['log_td_errors']))

        log_errors_l = locals['log_errors_l']  # Huber loss of td errors
        log_errors_l.append(np.mean(locals['log_errors']))

        log_gradients_l = locals['log_gradients']
        log_grad_ratio_l = locals['log_grad_ratio_l']

        for grad, var in log_gradients_l:
            grad_step = np.linalg.norm(grad*-locals['l_rate'])
            var_norm = np.linalg.norm(var)
            if var_norm > 0:
                wg_ratio = grad_step / var_norm
                log_grad_ratio_l.append(wg_ratio)


        if done:

            rew100 = locals['mean_100ep_reward']
            rew50 = locals['mean_50ep_reward']
            rew10 = locals['mean_10ep_reward']

            logger = globals['logger']

            action_counter = sorted(collections.Counter(log_action_l).items())

            episode_q_t_selected = np.mean(log_q_t_selected_l) #1x1
            episode_q_t_targets = np.mean(log_q_t_targets_l) #1x1
            #episode_q_t = np.mean(log_q_t_l,axis=0) #1x2

            episode_td_errors = np.mean(log_td_errors_l) #1x1
            episode_errors = np.mean(log_errors_l) # 1x1


            mean_wg_ratio = np.mean(log_grad_ratio_l) if len(log_grad_ratio_l)>0 else 0
            median_wg_ratio = np.median(log_grad_ratio_l) if len(log_grad_ratio_l)>0 else 0
            max_wg_ratio = np.max(log_grad_ratio_l) if len(log_grad_ratio_l)>0 else 0
            min_wg_ratio = np.min(log_grad_ratio_l) if len(log_grad_ratio_l)>0 else 0


            rew = locals['rew']

            logger.record_tabular("Tmp File", locals['td'])
            logger.record_tabular("Episode Rewards", rew)
            logger.record_tabular("Actions Count", action_counter)
            logger.record_tabular("Mean selected Q",episode_q_t_selected)
            logger.record_tabular("Mean selected target Q",episode_q_t_targets)
            #logger.record_tabular("Mean Action Q", episode_q_t)
            logger.record_tabular("Mean TD Error", episode_td_errors)
            logger.record_tabular("Mean Huber Error", episode_errors)
            logger.record_tabular("Var/Grad *-lr Mean ratio", mean_wg_ratio)
            logger.record_tabular("Var/Grad *-lr Median ratio", median_wg_ratio)
            logger.record_tabular("Var/Grad *-lr Max ratio", max_wg_ratio)
            logger.record_tabular("Var/Grad *-lr Min ratio", min_wg_ratio)
            logger.dump_tabular()


            if num_episodes % 100 == 0:
                #Write log




                path = locals['train_file_path']

                print("Writing episode {} log to ".format(num_episodes), path)

                with open(path, 'a') as f:

                    action_count_header = ['action_count{}'.format(i) for i in range(len(action_counter))]
                    #action_q_header = ['mean_action_q{}'.format(i) for i in range(len(episode_q_t.tolist()))]

                    headers = ['episode','steps','reward','reward100','reward50','reward10','mean_s_q','mean_t_q','mean_td_error','mean_h_error']
                    #headers = headers + action_q_header+action_count_header
                    headers = headers  + action_count_header +['mean_wg'] +['median_wg'] +['max_wg'] +['min_wg']

                    steps = locals['t']

                    action_counts = [c for i,c in action_counter]
                    #actions_qs = [q for q in episode_q_t.tolist()]

                    #output_list = [num_episodes]+[steps]+[rew100]+[rew50]+[rew10]+[episode_q_t_selected]+[episode_q_t_targets]+[episode_td_errors]+[episode_errors]+ actions_qs+action_counts
                    output_list = [num_episodes] + [steps] + [rew] +[rew100] + [rew50] + [rew10] + [episode_q_t_selected] + [
                        episode_q_t_targets] + [episode_td_errors] + [episode_errors] + action_counts +[mean_wg_ratio]+[median_wg_ratio]+[max_wg_ratio]+[min_wg_ratio]
                    print(headers)
                    print(output_list)
                    w = csv.writer(f)

                    if os.stat(path).st_size == 0:
                        w.writerow(headers)

                    w.writerow(output_list)

    return False



def main():
    #TODO :Test the prediction code on both environments




    #TODO: Try larger lr, try mpc guidance, try hard dataset (create ne for large trainset ... ) Note that naive has small error, means most of its rewards are 0 ... (unlike toy environment)
    #TODO: tehst mpc guidance
    #TODO: ask jan : step size, hard data set
    #TODO: prediction code (saliency, t-sne, value func on simple environment)
    #TODO: Cunt the amount of episodes in each set..  for paper
    # try out stuff  different sets, maybe different mpc where does not favour 0 action ... !
    #try out  slow taining upwards from each set hadest to easier ...
    # try gigantic replay buffer with larger set noticed that average sampled rewards  stabiliye if replaz buffer big, mazbe show example fomr old data or make experiment and extract batch of rewards
    # discourage adding 0 action to the replay buffer  alway take set where it started to struggle to see if it improves on more difficult set


    np.random.seed(1)
    random.seed(1)

    data_path = "/local/habichta/data_C_int/"
    mask_path = "/local/habichta/img_C/cavriglia_skymask256.png"
    img_path = "/local/habichta/img_C/"
    train_set_path = "train_list.out"

    channels = 3
    seq_length = 3
    img_size = 84
    seq_stride = 9


    env =   RealCloudEnvironment(data_path,img_path,train_set_path, image_size=img_size,
                 sequence_length=seq_length, sequence_stride=seq_stride, action_nr=7, action_type=1, ramp_step=0.1, episode_length_train=200,
                file="rl_data_sp.csv",load_train_episodes="ep100_200.pkl",mask_path=mask_path,sample_training_episodes=None,exploration_follow="IRR",start_exploration_deviation=0.2,clip_irradiance=False)


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
        lr=2.5e-4,
        max_timesteps=4500000,
        buffer_size=100000,
        initial_exploration=1.0,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=100000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=logger_callback,
        load_cpk="cloud_model400.pkl",
        mpc_guidance=0.25
        #"/media/nox/OS/Linux/Documents/Masterarbeit/simple_rl/model1/model"
    )
    act.save("cloud_model.pkl")


if __name__ == '__main__':
    main()

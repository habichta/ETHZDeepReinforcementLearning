import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import pandas as pd
import signal
import sys

from shutil import copyfile
from distutils.dir_util import copy_tree
import cv2
from scipy import misc




class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpu=16):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = deepq.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)


def load(path, num_cpu=16):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle
    num_cpu: int
        number of cpus to use for executing the policy

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path, num_cpu=num_cpu)


def learn(train_file_path,
          env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          initial_exploration=1.0,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          reward_priority=False,
          mpc_guidance=None,
          num_cpu=16,
          param_noise=False,
          callback=None,
          load_cpk=None,
          neutral_action_limit=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    num_cpu: int
        number of cpus to use for training
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)


    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha,neutral_action_limit=neutral_action_limit)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size,neutral_action_limit=neutral_action_limit)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=initial_exploration,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    if load_cpk:
        print('Loading model ', load_cpk)

        if load_cpk.endswith('pkl'):
            with open(load_cpk, "rb") as f:
                model_data, act_params = dill.load(f)

            with tempfile.TemporaryDirectory() as td:
                arc_path = os.path.join(td, "packed.zip")
                with open(arc_path, "wb") as f:
                    f.write(model_data)
                zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                U.load_state(os.path.join(td, "model"))

        else:

            U.load_state(load_cpk)






    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    episode_start = True

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        print(model_file)

        log_training_started = False

        def signal_handler(*args):
            cwd = os.getcwd()
            to = os.path.join(cwd,'model')
            print("Process interrupted, Copying ",td, 'to', to)
            copy_tree(td, to)

            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)  # Or whatever signal



        log_q_t_selected_l, log_q_t_targets_l_l, log_q_t_l, log_action_l, log_td_errors_l,log_errors_l,log_grad_ratio_l =[],[],[],[],[],[],[]
        for t in range(max_timesteps):

            log_trained = False

            #Logging locals
            l_rate,log_q_t_selected, log_q_t_targets_l, log_q_t,log_td_errors,log_errors,log_gradients=lr,-1,-1,-1,-1,-1,[]


            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            #add mpc guidance here, overwrite act action, set update_eps to 0 make custom exploration


            if mpc_guidance:

                if np.random.uniform(0.0,1.0) < update_eps:
                    action = env.mpc_exploration(mpc_guidance)
                else:
                    action = act(np.array(obs)[None], update_eps=0, **kwargs)[0]
            else:
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]


            reset = False
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.


            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                log_training_started=True
                log_trained = True

                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors,log_q_t_selected,log_q_t_targets,log_q_t,log_errors,gradients,batch_rewards = train(obses_t, actions, rewards, obses_tp1, dones, weights)


                log_td_errors = td_errors
                log_gradients = gradients



                if prioritized_replay:
                    if reward_priority:
                        new_priorities = np.abs(batch_rewards) + prioritized_replay_eps
                    else:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_500ep_reward = round(np.mean(episode_rewards[-501:-1]), 1)
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_50ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)

            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 500 episode reward", mean_500ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

            if callback is not None:
                if callback(locals(), globals()):
                    break

            if not done:
                episode_start=False
            if done:
                episode_start=True
                log_q_t_selected_l, log_q_t_targets_l_l, log_q_t_l, log_action_l, log_td_errors_l, log_errors_l, log_grad_ratio_l = [], [], [], [], [], [], []


        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    return ActWrapper(act, act_params)




def test(load_cpk,
          env,
          episode_n,
          q_func,
          result_callback,  #list of ci, list of episodes
          test_file_path='test_log.csv',
          print_freq=1,
          num_cpu=16,
          log_callback=None):




    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)


    if load_cpk is not None:

        if load_cpk.endswith('.pkl'):
            act_w = ActWrapper.load(load_cpk)
            act = act_w._act
        else:
            sess = U.make_session(num_cpu=num_cpu)
            sess.__enter__()
            act = deepq.build_act(make_obs_ph, q_func, env.action_space.n, scope="deepq", reuse=None)
            U.initialize()
            U.load_state(load_cpk)

    else:

        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        act = deepq.build_act(make_obs_ph, q_func, env.action_space.n, scope="deepq", reuse=None)
        U.initialize()






    cumulative_episode_rewards=[]
    ci_list =[]
    ep_list =[]


    ep_17,ep_100,ep_200,ep_300,ep_400,ep_500,ep_600 = [],[],[],[],[],[],[]



    for e in range(episode_n):

        obs = env.reset()
        done = False
        episode_rewards = []
        log_action_l = []
        num_episodes = e
        while not done:

            # Logging locals
            action = act(np.array(obs)[None], update_eps=0)[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            obs = new_obs
            episode_rewards.append(rew)


            if done:
                cumulative_episode_rewards.extend(episode_rewards)


            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("episode", num_episodes)
                logger.record_tabular("episodes in set", env.episode_n)
                logger.record_tabular("episode_reward_sum", np.sum(episode_rewards))
                logger.record_tabular("episode_reward_mean",np.mean(episode_rewards))
                logger.record_tabular("episodes_cumulative_reward_sum", np.sum(cumulative_episode_rewards))
                logger.record_tabular("episodes_cumulative_reward_mean", np.mean(cumulative_episode_rewards))
                logger.dump_tabular()


            if log_callback is not None:
                if log_callback(locals(), globals()):
                    break

            if done:
                episode_rewards,log_action_l=[],[]



        control_inputs_tuple = env.current_train_control_inputs
        current_episode = env.current_train_episode


        ci_list.append(control_inputs_tuple)
        ep_list.append(current_episode)



    result_callback(ci_list,ep_list,locals(),globals())













class ActWrapper_sal(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params


    @staticmethod
    def load(path, num_cpu=16):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = deepq.build_act_sal(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()

        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params),sess

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)






def learn_sal(train_file_path,
          env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          initial_exploration=1.0,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          reward_priority=False,
          mpc_guidance=None,
          num_cpu=16,
          param_noise=False,
          callback=None,
          load_cpk=None,
          neutral_action_limit=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    num_cpu: int
        number of cpus to use for training
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)


    act_sal, train, update_target, debug = deepq.build_train_sal(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha,neutral_action_limit=neutral_action_limit)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size,neutral_action_limit=neutral_action_limit)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=initial_exploration,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    if load_cpk:
        print('Loading model ', load_cpk)

        if load_cpk.endswith('pkl'):
            with open(load_cpk, "rb") as f:
                model_data, act_params = dill.load(f)

            with tempfile.TemporaryDirectory() as td:
                arc_path = os.path.join(td, "packed.zip")
                with open(arc_path, "wb") as f:
                    f.write(model_data)
                zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
                U.load_state(os.path.join(td, "model"))

        else:

            U.load_state(load_cpk)






    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    episode_start = True

    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        print(model_file)

        log_training_started = False

        def signal_handler(*args):
            cwd = os.getcwd()
            to = os.path.join(cwd,'model')
            print("Process interrupted, Copying ",td, 'to', to)
            copy_tree(td, to)

            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)  # Or whatever signal



        log_q_t_selected_l, log_q_t_targets_l_l, log_q_t_l, log_action_l, log_td_errors_l,log_errors_l,log_grad_ratio_l =[],[],[],[],[],[],[]
        for t in range(max_timesteps):

            log_trained = False

            #Logging locals
            l_rate,log_q_t_selected, log_q_t_targets_l, log_q_t,log_td_errors,log_errors,log_gradients=lr,-1,-1,-1,-1,-1,[]


            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            #add mpc guidance here, overwrite act action, set update_eps to 0 make custom exploration


            if mpc_guidance:

                if np.random.uniform(0.0,1.0) < update_eps:
                    action = env.mpc_exploration(mpc_guidance)
                else:
                    action,_,_,_,_,_ = act_sal(np.array(obs)[None], update_eps=0, **kwargs)
                    action = action[0]
            else:
                action,_,_,_,_,_ = act_sal(np.array(obs)[None], update_eps=update_eps, **kwargs)
                action = action[0]

            reset = False
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.


            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                log_training_started=True
                log_trained = True

                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors,log_q_t_selected,log_q_t_targets,log_q_t,log_errors,gradients,batch_rewards = train(obses_t, actions, rewards, obses_tp1, dones, weights)


                log_td_errors = td_errors
                log_gradients = gradients



                if prioritized_replay:
                    if reward_priority:
                        new_priorities = np.abs(batch_rewards) + prioritized_replay_eps
                    else:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_500ep_reward = round(np.mean(episode_rewards[-501:-1]), 1)
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_50ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)

            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 500 episode reward", mean_500ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

            if callback is not None:
                if callback(locals(), globals()):
                    break

            if not done:
                episode_start=False
            if done:
                episode_start=True
                log_q_t_selected_l, log_q_t_targets_l_l, log_q_t_l, log_action_l, log_td_errors_l, log_errors_l, log_grad_ratio_l = [], [], [], [], [], [], []


        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    return ActWrapper_sal(act_sal, act_params)




def saliency(grads,mask):


    grads[mask] = 0.0

    image_2d = np.sum(np.abs(grads), axis=2)
    #image_2d = np.abs(grads)
    #image_2d = np.abs(grads[:,:,1])

    vmax = np.percentile(image_2d, 99)
    vmin = np.percentile(image_2d,70)

    print(vmax,vmin)
    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)




def test_sal(load_cpk,
          env,
          episode_n,
          q_func,
          result_callback,  #list of ci, list of episodes
          test_file_path='test_log.csv',
          print_freq=1,
          num_cpu=16,
          log_callback=None):




    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)


    if load_cpk is not None:

        if load_cpk.endswith('.pkl'):
            act_w,sess = ActWrapper_sal.load(load_cpk)
            act = act_w._act
        else:
            sess = U.make_session(num_cpu=num_cpu)
            sess.__enter__()
            act = deepq.build_act_sal(make_obs_ph, q_func, env.action_space.n, scope="deepq", reuse=None)
            U.initialize()
            U.load_state(load_cpk)

    else:

        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        act = deepq.build_act_sal(make_obs_ph, q_func, env.action_space.n, scope="deepq", reuse=None)
        U.initialize()




    #gradient_saliency = saliency.GradientSaliency(sess.graph,sess)


    cumulative_episode_rewards=[]
    ci_list =[]
    ep_list =[]


    ep_17,ep_100,ep_200,ep_300,ep_400,ep_500,ep_600 = [],[],[],[],[],[],[]

    mask = misc.imread("/media/nox/OS/Linux/Documents/Masterarbeit/data/Daten/img_C/cavriglia_skymask256.png")==0
    mask = misc.imresize(mask, [84,84, 3])
    mask = np.repeat(mask,3,axis=2)>0

    action_df = []


    col_df = ['action','action_val0','action_val1','action_val2','val']


    for e in range(episode_n):

        obs = env.reset()
        done = False
        episode_rewards = []
        log_action_l = []
        num_episodes = e
        i = 0
        while not done:
            i = i+1

            # Logging locals
            action,state_score, action_scores, action_salience, state_salience,salience_image = act(np.array(obs)[None], update_eps=0)


            action_scores = action_scores[0]
            state_score = state_score[0]

            print(action,action_scores,state_score)

            data = np.concatenate((action,action_scores,state_score))

            action = action[0]
            action_df.append(data)

            m = salience_image[0][:,:,6:9].copy()


            asal = np.repeat(saliency(action_salience[0],mask)[:,:,np.newaxis],3,axis=2).copy()
            vsal = np.repeat(saliency(state_salience[0],mask)[:,:,np.newaxis],3,axis=2).copy()


            #cv2.addWeighted(asal, 0.7, m, 1 - 0.7,0, m)
            asal[:,:,[0,1]]=0.0
            vsal[:,:,[0,1]]=0.0
            ma = m + 0.9*asal
            mv = m + 0.9 * vsal


            cv2.imwrite('/home/nox/Desktop/img/val/'+str(i)+'.jpg',np.uint8(255*mv.copy()))
            cv2.imwrite('/home/nox/Desktop/img/adv/' + str(i) + '.jpg', np.uint8(255*ma.copy()))
            cv2.imwrite('/home/nox/Desktop/img/norm/' + str(i) + '.jpg', np.uint8(255*m.copy()))

            cv2.imshow('',np.concatenate((ma,mv),axis=1))
            #cv2.imshow('',overlay(asal,salience_image[0][:,:,6:9]))
            #cv2.imshow("",m)
            cv2.waitKey(10)

            #np.max(np.squeeze(action_salience[0]), axis=2, keepdims=True)
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            obs = new_obs
            episode_rewards.append(rew)


            if done:
                cumulative_episode_rewards.extend(episode_rewards)


            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("episode", num_episodes)
                logger.record_tabular("episodes in set", env.episode_n)
                logger.record_tabular("episode_reward_sum", np.sum(episode_rewards))
                logger.record_tabular("episode_reward_mean",np.mean(episode_rewards))
                logger.record_tabular("episodes_cumulative_reward_sum", np.sum(cumulative_episode_rewards))
                logger.record_tabular("episodes_cumulative_reward_mean", np.mean(cumulative_episode_rewards))
                logger.dump_tabular()


            if log_callback is not None:
                if log_callback(locals(), globals()):
                    break

            if done:
                episode_rewards,log_action_l=[],[]



        control_inputs_tuple = env.current_train_control_inputs
        current_episode = env.current_train_episode


        ci_list.append(control_inputs_tuple)
        ep_list.append(current_episode)





    df = pd.DataFrame(data=action_df,columns=col_df)
    df.to_csv('/home/nox/Desktop/img/action.csv')
    result_callback(ci_list,ep_list,locals(),globals())







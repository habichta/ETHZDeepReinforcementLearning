import sys

import numpy as np
import pandas as pd
import random
import os
from scipy import misc
import pickle
import cv2


#TODO: note gradient norm is clipped by baseline at 10

class RealCloudEnvironment():
    def __init__(self, data_path,img_path,train_set_path, image_size=84,
                 sequence_length=4, sequence_stride=9, action_nr=7, action_type=1,adapt_step_size=False, ramp_step=0.1, episode_length_train=None,
                  file="rl_data_sp.csv",load_train_episodes=None,mask_path=None,sample_training_episodes=None,exploration_follow="IRR",start_exploration_deviation=0.2,clip_irradiance=False):


        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.episode_length_train = episode_length_train
        self.ramp_step = ramp_step
        self.image_size = image_size
        self.load_train_episodes = load_train_episodes
        self.mask_path = mask_path
        self.sample_training_episodes = sample_training_episodes
        self.start_exploration_deviation = start_exploration_deviation
        self.exploration_follow=exploration_follow
        self.adapt_step_size=adapt_step_size
        self.clip_irradiance = clip_irradiance


        self.observation_space = self.ObservationSpace(
            (image_size * image_size * sequence_length * 3 + sequence_length + 1, 1))

        # self.observation_space

        self.action_space = self.ActionSpace(action_type, action_nr, ramp_step,adapt_step_size)


        if self.mask_path:
            self.mask=misc.imread(self.mask_path)==0 #255 and 0 values
        else:
            self.mask=None


        self.file_path = os.path.join(data_path, file)
        self.img_path = img_path


        # Episodes:
        self.train_episodes = self.__create_episodes(train_set_path=train_set_path)
        self.nr_train_episodes = len(self.train_episodes)
        self.temp_train_episodes = list(self.train_episodes)


        # Training globals
        self.current_episode_train_step_pointer = None
        self.current_episode_train = None
        self.current_episode_train_control_input_values = []
        self.start_date = None
        self.end_date = None

    @property
    def current_train_episode(self):
        return self.current_episode_train


    @property
    def current_train_control_inputs(self):
        return self.current_episode_train_control_input_values

    @property
    def episode_n(self):
        return self.nr_train_episodes
    @property
    def episode_id(self):
        return self.start_date
    @property
    def episode_end_id(self):
        return self.end_date

    def reset(self):

        print("Resetting environment...")
        if not self.temp_train_episodes:
            print("Epoch finished...")
            # When all trianing episodes have been sampled at least once, renew the list, start again
            self.temp_train_episodes = list(self.train_episodes)
        print("Sampling episode...")
        # Sample a random episode from the train_episodes list, delete it from list so that it is not sampled in this epoch again
        self.current_episode_train = self.temp_train_episodes.pop(
            random.randrange(len(self.temp_train_episodes)))  # sample episode and remove from temporary list

        print("Episode (from/to): ", str(self.current_episode_train.index[0]),
              str(self.current_episode_train.index[-1]))

        print("Samples in episode:", len(self.current_episode_train))

        # get index from current eppisode (Datetime)
        index = self.current_episode_train.index.tolist()

        self.start_date =index[0]
        self.end_date = index[-1]


        # Create index  for smples depending on image sequence length and stride
        self.train_episode_samples = [index[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride]
                                      for i in
                                      range(len(index) - (self.sequence_length - 1) * self.sequence_stride)]



        # Set pointer to the current sample, advanced by step()
        self.current_episode_train_step_pointer = 0

        # Get first sample index, list of timestamps of the images and irradiance data
        first_state_index = self.train_episode_samples[self.current_episode_train_step_pointer]

        # Load actual data given the timestamps
        current_state = self.current_episode_train.loc[first_state_index]

        # list of image_names
        images_names = current_state['img_name'].values
        # create paths to images of that sample
        image_paths = [os.path.join(self.img_path, name) for name in images_names]

        # Initialize irradiance and control input
        curr_irr =np.array(current_state["irr"].values)
        curr_mpc = np.array(current_state["mpc"].values)

        #MPC follow : current_control_input = current_mpc[-1]
        #Random:
        if self.exploration_follow == "IRR":
            curr_ci =  curr_irr[-1]

        elif self.exploration_follow == "MPC":
            curr_ci = curr_mpc[-1]

        else:
            raise ValueError("Choose correct exploration follow: IRR or MPC")

        if self.start_exploration_deviation:
            curr_ci = curr_ci+np.float32(np.random.uniform(-self.start_exploration_deviation,self.start_exploration_deviation)) # at least some different steps in beginning of episodes
        #Check:
        if curr_ci< 0.0:
            curr_ci = 0.0

        #current_control_input = np.random.uniform(200.0,800.0)

        # Reset list that stores all controlinputs for an episode and append first control input
        current_timestamp = current_state.index[-1]
        self.current_episode_train_control_input_values = []
        self.current_episode_train_control_input_values.append(
            (curr_ci, current_timestamp))  # add tuple with control input and timestamp

        # Decode jpeg images and preprocess
        image_tensor = self.__decode_image(image_paths)

        env_obs = np.concatenate([image_tensor.ravel(), curr_irr, np.reshape(curr_ci, (1))]).astype(np.float16)[:, None]

        """
        cv2.imshow('next_state_image_32', np.uint8(np.reshape(env_obs[0:-3], (84, 84, 6))[:, :, 3:6]))
        cv2.waitKey(50)
        """

        return env_obs

    def step(self, action):

        # Update step variable
        current_step = self.current_episode_train_step_pointer
        self.current_episode_train_step_pointer += 1  # next step to get data of next state
        next_step = self.current_episode_train_step_pointer

        # get state data
        current_state = self.current_episode_train.loc[self.train_episode_samples[current_step]]

        next_state = self.current_episode_train.loc[self.train_episode_samples[next_step]]  # data of next state

        next_irr = np.array(next_state["irr"].values)  # irradiance in next step batch x 1
        curr_irr = np.array(current_state["irr"].values)

        current_control_input = self.current_episode_train_control_input_values[-1][
            0]  # get last control_input from list

        # calculate the next controlinput given the current input and the time difference + ramp between current and next state

        next_ci, reward = self.action_space.calculate_step(action=action,next_irr=next_irr[-1],curr_irr=curr_irr[-1],current_ci=current_control_input,curr_index=current_state.index.values[
            -1],next_index=next_state.index.values[-1])

        # Update control input list
        next_timestamp = next_state.index[-1]
        self.current_episode_train_control_input_values.append(
            (next_ci, next_timestamp))  # Add next ocntrol input value

        # done: whether the next state is the last of the episode. Z.b. end of day
        done = bool(next_state.iloc[-1]["done"])

        # Get images of next state
        images_names = next_state['img_name'].values
        image_paths = [os.path.join(self.img_path, name) for name in images_names]
        next_image_tensor = self.__decode_image(image_paths)

        next_env_obs = np.concatenate([next_image_tensor.ravel(), next_irr, np.reshape(next_ci, (1))]).astype(np.float16)[:,None]



        #DEBUG:  #######################################################################################################
        #Show both images of next state
        """
        cv2.imshow('next_state_image_32',
                   np.uint8(np.concatenate((np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 0:3],
                                            np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 3:6]), axis=1)))
        cv2.waitKey(5)

        print(next_timestamp, "   reward:", reward, "   next_irr:", '{0:.8f}'.format(next_irr[-1]), "   next_ci:",
              '{0:.8f}'.format(next_ci), "   action:", action)
        cv2.imshow('next_state_image_32',
                   np.uint8((np.reshape(next_env_obs[0:-3], (128, 128, 6))[:, :, 0] -
                             np.reshape(next_env_obs[0:-3], (128, 128, 6))[:, :, 3])))
        cv2.waitKey(50)
        
       
        
        #Show difference of both next state images
        cv2.imshow('next_state_image_32',
                   np.uint8((np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 0] -
                                            np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 3])))
       

        images_names = current_state['img_name'].values
        image_paths = [os.path.join(self.img_path, name) for name in images_names]
        current_image_tensor = self.__decode_image(image_paths)

        
        #Show both current image nad next state image
        cv2.imshow('next_state_image_32',
                   np.uint8(np.concatenate((current_image_tensor[:,:,3:6],
                                            np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 3:6]), axis=0)))
        
        #Show difference between current state image and next state image. Check if there is at least some difference between them in 84x84
        cv2.imshow('next_state_image_32',
                   np.uint8((current_image_tensor[:, :, 2]-np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 2]), axis=0))

        if done:
    
            irr_list, mpc_list, ci_list, index_list = [], [], [], []
            # column_list = ["irr", "mpc", "ci"]
    
            for t in self.current_train_control_inputs:
                index_list.append(t[1])
                ci_list.append(t[0])
    
            ci_df = pd.DataFrame(data=ci_list, index=index_list, columns=["ci"])
    
            irrmpc_df = self.current_episode_train.loc[ci_df.index]
    
            data_df = pd.concat([ci_df, irrmpc_df], axis=1)
    
            #data_df[["ci", "irr", "mpc"]].plot()
    
            print(data_df)
    
            #plt.show()
        #######################################################################################################
        """

        return next_env_obs, reward, done,0  # return s',r,d



    def __decode_image(self, image_paths):
        #Node newer images are further back in terms of channel coordinates! 0:3 -> first image .... etc. the last iamge is in the last 3 channels
        image_np = np.concatenate([self.__preprocess_image(cv2.imread(image)) for image in image_paths], axis=2)
        return image_np

    def __preprocess_image(self, image):

        if self.mask_path:
            image[self.mask]=0.0
        image = misc.imresize(image, [self.image_size, self.image_size, 3])

        return image


    def __create_episodes(self, train_set_path):

        print("Environment: Loading rl_data file and datasets...")
        rl_pd = pd.DataFrame.from_csv(self.file_path).sort_index()

        #Divide mpc and irr by 1000 to normalize all values between 0 and 1 (more or less, since there is some irradiance >1000):
        rl_pd[['mpc','irr','cs']] = rl_pd[['mpc','irr','cs']]/1000.0





        if train_set_path:
            print("reading " + str(train_set_path))
            with open(str(train_set_path)) as f:
                self.train_list = sorted([os.path.basename(l).split('-', 1)[1] for l in f.read().splitlines()])
        else:
            self.train_list = None


        print("Creating episodes...")
        train_episodes = []


        if self.load_train_episodes:
            with open(self.load_train_episodes,'rb') as f:
                train_episodes = pickle.load(f)

        else:
            if self.train_list:
                for train_day_it in self.train_list:
                    td_pd = pd.DataFrame(rl_pd.loc[train_day_it])

                    if self.episode_length_train is None:  # 1 day = 1 episode
                        done_pd = np.zeros(len(td_pd.index)).astype(int)
                        done_pd[-1] = 1
                        td_pd["done"] = done_pd
                        train_episodes.append(td_pd)
                    else:

                        for g, episode in td_pd.groupby(np.arange(len(td_pd)) // self.episode_length_train):
                            episode_df = pd.DataFrame(episode)
                            done_pd = np.zeros(len(episode_df.index)).astype(int)
                            done_pd[-1] = 1
                            episode_df["done"] = done_pd
                            train_episodes.append(episode_df)



        print("Episodes in Set:" ,len(train_episodes))

        train_episodes_filtered = [te for te in train_episodes if self.filter_episodes(te)] # filter out too small episodes. at 1 step

        train_episodes_final = []
        if self.clip_irradiance: #changes irradiance to 1.0 or 0.0. 1.0 if irradiance is larger than 70% of clear sky model

            for e_pd in train_episodes_filtered:
                e_pd['irr'] = np.where(e_pd['cs'] * 0.7 < e_pd['irr'], 1.0, 0.0)
                train_episodes_final.append(e_pd)
        else:
            train_episodes_final=train_episodes_filtered



        if self.sample_training_episodes:
            train_episodes_final = np.random.choice(train_episodes_final,size=self.sample_training_episodes)

        print("Episodes in Set (after filter and sampling):", len(train_episodes))

        return train_episodes_final

    def filter_episodes(self, df):
        keep = True

        df['tvalue'] = df.index
        df['delta'] = (df['tvalue'] - df['tvalue'].shift()).fillna(0)

        if np.max(df['delta']/ np.timedelta64(1,'m')) > 14.0: # check there are not too large  time differences between samples
            keep = False

        if len(df) < self.sequence_length*self.sequence_stride+1: # some episodes can be too small since not for all days sample % episode_size == 0 !
            keep = False

        return keep





    def mpc_exploration(self, mpc_prob=0.5):
        """
        env.reset needs to be called first. Create exploration that follows MPC in trianing set to a certain degree
        :param mpc_prob: Probability of taking action that gets closest to mpc (other actions will be chosen with probability (1-p)/(num actions-1)
        :param num_actions: nr actions
        :return: action to choose (integer)
        """
        # Get next state
        current_step = self.current_episode_train_step_pointer
        next_step = self.current_episode_train_step_pointer + 1

        current_state = self.current_episode_train.loc[self.train_episode_samples[current_step]]
        next_state = self.current_episode_train.loc[self.train_episode_samples[next_step]]

        next_irr = np.array(next_state["irr"].values)
        curr_irr = np.array(current_state["irr"].values)


        current_control_input = self.current_episode_train_control_input_values[-1][
            0]  # get last control_input from list
        mpc = np.array(next_state["mpc"].values)[-1]

        control_inputs = list()
        for a in range(self.action_space.n):
            next_ci, _ = self.action_space.calculate_step(action=a, next_irr=next_irr[-1],
                                                               curr_irr=curr_irr[-1], current_ci=current_control_input,
                                                               curr_index=current_state.index.values[
                                                                   -1], next_index=next_state.index.values[-1])



            control_inputs.append(abs(next_ci - mpc))




        #best_action = np.argmin(control_inputs[1:])+1 #do not take 0 action into account, only favour non zero
        best_action = np.argmin(control_inputs)



        action_array = np.arange(0,self.action_space.n, 1)
        normal_action_weight = (1 - mpc_prob) / (self.action_space.n - 1)
        action_weights = np.ones(self.action_space.n) * normal_action_weight
        action_weights[best_action] = mpc_prob

        action = np.random.choice(action_array, replace=False, p=action_weights)

        return action


    class ActionSpace():
        def __init__(self, type=0, action_nr=2, ramp_step=0.1,adapt_step_size=False):
            self.n = action_nr
            self.type = type
            self.ramp_step = ramp_step
            self.adapt_step_size = adapt_step_size

        def calculate_step(self, action, next_irr, curr_irr, current_ci,curr_index,next_index):
            # return next ci given action

            ramp_step_s = self.ramp_step/60.0
            if self.adapt_step_size:
                stretch_factor = (next_index - curr_index) / np.timedelta64(1, 's')
                print("stretch_factor:",stretch_factor)
            else:
                stretch_factor = 7.0  # median difference between states in 7 seconds



            if self.type == 0:
                if action == 0:  # upper target
                    next_ci = float(curr_irr)
                elif action == 1:  # upper target
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor
                elif action == 2:  # lower target
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor)
                elif action == 3:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor / 2
                elif action == 4:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor / 2)
                elif action == 5:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor / 4
                elif action == 6:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor / 4)
                else:
                    raise ValueError('Illegal action')


            elif self.type == 1:
                if action == 0:
                    diff = curr_irr - current_ci
                    ramp = np.abs(diff) > ramp_step_s*stretch_factor
                    if ramp:
                        next_ci = current_ci + np.sign(diff) * ramp_step_s*stretch_factor
                    else:
                        next_ci = curr_irr
                elif action == 1:  # upper target
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor
                elif action == 2:  # lower target
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor)
                elif action == 3:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor/2
                elif action == 4:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor/2)
                elif action == 5:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor / 4
                elif action == 6:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor / 4)
                else:
                    raise ValueError('Illegal action')

            elif self.type == 2:
                if action == 0:
                    diff = next_irr - current_ci
                    ramp = np.abs(diff) > ramp_step_s*stretch_factor
                    if ramp:
                        next_ci = current_ci + np.sign(diff) * ramp_step_s*stretch_factor
                    else:
                        next_ci = next_irr
                elif action == 1:  # upper target
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor
                elif action == 2:  # lower target
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor)
                elif action == 3:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor/2
                elif action == 4:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor/2)
                elif action == 5:
                    next_ci = float(current_ci) + ramp_step_s*stretch_factor / 4
                elif action == 6:
                    next_ci = np.maximum(0.0, current_ci - ramp_step_s*stretch_factor / 4)
                else:
                    raise ValueError('Illegal action')

            elif self.type == -1:
                # naive policy with curr irr
                diff = curr_irr - current_ci
                ramp = np.abs(diff) > ramp_step_s*stretch_factor
                if ramp:
                    next_ci = current_ci + np.sign(diff) * ramp_step_s*stretch_factor
                else:
                    next_ci = curr_irr
            else:

                raise ValueError('Illegal Action Set')

            reward = np.maximum(-np.abs(next_ci - next_irr).squeeze(),-3.0) # clip reward against outliers/errors. should not reach a level of -3.0



            return next_ci, reward

    class ObservationSpace():
        def __init__(self, shape):
            self.shape = shape
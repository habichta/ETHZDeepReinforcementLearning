import sys

sys.path.append("/home/dladmin/Documents/arthurma/shared/dlabb")
sys.path.append("/home/habichta/dlabb")
sys.path.append("/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/")
import numpy as np
import pandas as pd
import random
import os
from scipy import misc
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
from scipy import ndimage
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as arp
import pickle


class Environment():
    def __init__(self, train_set_path, test_set_path, solar_station=ac.ABB_Solarstation.C, image_size=84,
                 sequence_length=2, sequence_stride=9, actions=7, max_ramp_per_m=100, episode_length_train=None,episode_length_test=None,
                 action_space=1, file="rl_data.csv",load_train_episodes=None,load_test_episodes=None,mask_path=None,divide_image_values=None,sample_training_episodes=None,exploration_follow="IRR",start_exploration_deviation=100,reward_type=1):
        self.actions = actions
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.episode_length_train = episode_length_train
        self.episode_length_test = episode_length_test
        self.max_ramp_per_m = max_ramp_per_m
        self.image_size = image_size
        self.action_space = action_space
        self.load_train_episodes = load_train_episodes
        self.load_test_episodes = load_test_episodes
        self.mask_path = mask_path
        self.divide_image_values = divide_image_values
        self.sample_training_episodes = sample_training_episodes
        self.start_exploration_deviation = start_exploration_deviation
        self.exploration_follow=exploration_follow
        self.reward_type = reward_type

        if self.mask_path:
            self.mask=misc.imread(self.mask_path)==0 #255 and 0 values
        else:
            self.mask=None


        if solar_station == ac.ABB_Solarstation.C:
            self.file_path = os.path.join(ac.c_int_data_path, file)
            self.img_path = ac.c_img_path
        elif solar_station == ac.ABB_Solarstation.MS:
            self.file_path = os.path.join(ac.ms_int_data_path, file)
            self.img_path = ac.ms_img_path
        else:
            raise ValueError("Illegal solar station")

        # Episodes:
        self.train_episodes, self.test_episodes = self.__create_episodes(train_set_path=train_set_path,
                                                                         test_set_path=test_set_path)
        self.nr_train_episodes = len(self.train_episodes)
        self.nr_test_episodes = len(self.test_episodes)
        self.temp_train_episodes = list(self.train_episodes)
        self.temp_test_episodes = list(self.test_episodes)

        # Training globals
        self.current_episode_train_step_pointer = None
        self.current_episode_train = None
        self.current_episode_train_control_input_values = list()

        # Training globals
        self.current_episode_test_step_pointer = None
        self.current_episode_test = None
        self.current_episode_test_control_input_values = list()

    @property
    def current_train_episode(self):
        return self.current_episode_train

    @property
    def current_test_episode(self):
        return self.current_episode_test

    @property
    def current_train_control_inputs(self):
        return self.current_episode_train_control_input_values

    @property
    def current_test_control_inputs(self):
        return self.current_episode_test_control_input_values

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

        # Create index  for smples depending on image sequence length and stride
        self.train_episode_samples = [index[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride]
                                      for i in
                                      range(len(index) - (self.sequence_length - 1) * self.sequence_stride)]

        abort = False
        if (len(self.train_episode_samples) > 1):
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
            current_irradiance =np.array(current_state["irr"].values)
            current_mpc = np.array(current_state["mpc"].values)

            #MPC follow : current_control_input = current_mpc[-1]
            #Random:
            if self.exploration_follow == "IRR":
                current_control_input = current_irradiance[-1]

            elif self.exploration_follow == "MPC":
                current_control_input = current_mpc[-1]

            else:
                raise ValueError("Choose correct exploration follow: IRR or MPC")

            if self.start_exploration_deviation:
                current_control_input = current_control_input+np.float32(np.random.randint(-self.start_exploration_deviation,self.start_exploration_deviation)) # at least some different steps in beginning of episodes



            #Check:
            if current_control_input < 0.0:
                current_control_input = 0.0
            #current_control_input = np.random.uniform(200.0,800.0)

            # Reset list that stores all controlinputs for an episode and append first control input
            current_timestamp = current_state.index[-1]
            self.current_episode_train_control_input_values = []
            self.current_episode_train_control_input_values.append(
                (current_control_input, current_timestamp))  # add tuple with control input and timestamp

            # Decode jpeg images and preprocess
            image_tensor = self.__decode_image(image_paths)

            # State:[image: z.b. 84x84x6 tensor, curr_irr float, curr_control_input float]
            first_state = np.array([image_tensor, current_irradiance, current_control_input])  # initial state
        else:
            first_state = None
            abort = True
            print("Episode size is too small, abort this episode")

        return first_state, abort

    def step(self, action):

        # Update step variable
        current_step = self.current_episode_train_step_pointer
        self.current_episode_train_step_pointer += 1  # next step to get data of next state
        next_step = self.current_episode_train_step_pointer

        # get state data
        current_state = self.current_episode_train.loc[self.train_episode_samples[current_step]]

        next_state = self.current_episode_train.loc[self.train_episode_samples[next_step]]  # data of next state

        next_irr = np.array(next_state["irr"].values)  # irradiance in next step batch x 1

        current_control_input = self.current_episode_train_control_input_values[-1][
            0]  # get last control_input from list

        # calculate the next controlinput given the current input and the time difference + ramp between current and next state



        if self.action_space == 1:
            next_control_input = self.__calculate_next_control_input_follow_irr(action, current_state, next_state,
                                                                                current_control_input)

        elif self.action_space == 2:
            next_control_input = self.__calculate_next_control_input_follow_irr_straight0(action, current_state, next_state,
                                                                                current_control_input)

        elif self.action_space == 3:
            next_control_input = self.__calculate_next_control_input_target_simple(action, current_state,
                                                                                   next_state,
                                                                                   current_control_input)

        else:
            next_control_input = self.__calculate_next_control_input(action, current_state, next_state,

                                                                     current_control_input)

        #TODO: remove?
        if next_control_input < 0:  # lower boound control input. Otherwise network may train on unrealistic experience of negative control input (there is no negative irradiance)
            next_control_input = 0

        # Update control input list
        next_timestamp = next_state.index[-1]
        self.current_episode_train_control_input_values.append(
            (next_control_input, next_timestamp))  # Add next ocntrol input value

        # reward is negative difference between next irr and next control input. Maximizing reward will reduce difference of irr and control input
        reward = self.__calculate_step_reward(next_irr[-1], next_control_input,action=action)


        #Clip reward during training! Since input data is noisy, there can be steps that are far larger than 7 seconds. This may lead to unnaturally low rewards
        #Could be a reason for divergence!
        reward = np.maximum(reward,-800.0) #set reward high enough such that it still allows most "normal values"

        # done: whether the next state is the last of the episode. Z.b. end of day
        done = next_state.iloc[-1]["done"]

        # Get images of next state
        images_names = next_state['img_name'].values
        image_paths = [os.path.join(self.img_path, name) for name in images_names]
        image_tensor = self.__decode_image(image_paths)

        return np.array([image_tensor, next_irr, next_control_input]), reward, done  # return s',r,d

    def test_reset(self):

        print("Resetting test environment...")
        if not self.temp_test_episodes:
            print("Epoch finished...")
            # When all trianing episodes have been sampled at least once, renew the list, start again
            self.temp_test_episodes = list(self.test_episodes)

        # Go along episodes in order
        self.current_episode_test = self.temp_test_episodes.pop()  # sample episode and remove from temporary list

        print("Episode (from/to): ", str(self.current_episode_test.index[0]),
              str(self.current_episode_test.index[-1]))
        # get index from current eppisode (Datetime)
        index = self.current_episode_test.index.tolist()

        # Create index  for smples depending on image sequence length and stride
        self.test_episode_samples = [index[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride]
                                     for i in
                                     range(len(index) - (self.sequence_length - 1) * self.sequence_stride)]

        abort = False
        if (len(self.test_episode_samples) > 1):  # at least one step should be possible so length must be at least 2
            # Set pointer to the current sample, advanced by step()
            self.current_episode_test_step_pointer = 0

            # Get first sample index, list of timestamps of the images and irradiance data
            first_state_index = self.test_episode_samples[self.current_episode_test_step_pointer]

            # Load actual data given the timestamps
            current_state = self.current_episode_test.loc[first_state_index]

            # list of image_names
            images_names = current_state['img_name'].values
            # create paths to images of that sample
            image_paths = [os.path.join(self.img_path, name) for name in images_names]

            # Initialize irradiance and control input

            current_irradiance = np.array(current_state["irr"].values)

            current_control_input = current_irradiance[-1]

            # Reset list that stores all controlinputs for an episode and append first controlinput
            current_timestamp = current_state.index[-1]
            self.current_episode_test_control_input_values = []
            self.current_episode_test_control_input_values.append((current_control_input, current_timestamp))

            # Decode jpeg images and preprocess
            image_tensor = self.__decode_image(image_paths)

            # State:[image: z.b. 84x84x6 tensor, curr_irr float, curr_control_input float]
            first_state = np.array([image_tensor, current_irradiance, current_control_input])
        else:
            first_state = None
            abort = True
            print("Episode size is too small, abort this episode")
        return first_state, abort

    def test_step(self, action):
        # Update step variable


        current_step = self.current_episode_test_step_pointer
        self.current_episode_test_step_pointer += 1  # next step to get data of next state
        next_step = self.current_episode_test_step_pointer

        # get state data
        current_state = self.current_episode_test.loc[self.test_episode_samples[current_step]]
        next_state = self.current_episode_test.loc[self.test_episode_samples[next_step]]  # data of next state
        next_irr = next_state["irr"].values  # irradiance in next step

        current_control_input = self.current_episode_test_control_input_values[
            -1][0]  # get last control_input from list

        # calculate the next controlinput given the current input and the time difference + ramp between current and next state

        if self.action_space == 1:
            next_control_input = self.__calculate_next_control_input_follow_irr(action, current_state, next_state,
                                                                                current_control_input)

        elif self.action_space == 2:
            next_control_input = self.__calculate_next_control_input_follow_irr_straight0(action, current_state, next_state,
                                                                                current_control_input)


        elif self.action_space==3:
            next_control_input = self.__calculate_next_control_input_target_simple(action, current_state,
                                                                                          next_state,
                                                                                          current_control_input)


        else:
            next_control_input = self.__calculate_next_control_input(action, current_state, next_state,
                                                                     current_control_input)

        if next_control_input < 0:  # lower boound control input. Otherwise network may train on unrealistic experience of negative control input (there is no negative irradiance)
            next_control_input = 0

        # Update control input list
        next_timestamp = next_state.index[-1]
        self.current_episode_test_control_input_values.append((next_control_input, next_timestamp))

        # reward is negative difference between next irr and next control input. Maximizing reward will reduce difference of irr and control input
        reward = self.__calculate_step_reward(next_irr[-1], next_control_input,action=action)

        # done: whether the next state is the last of the episode. Z.b. end of day
        done = next_state.iloc[-1]["done"]

        # Get images of next state
        images_names = next_state['img_name'].values
        image_paths = [os.path.join(self.img_path, name) for name in images_names]

        image_tensor = self.__decode_image(image_paths)

        return np.array([image_tensor, next_irr, next_control_input]), reward, done  # return s',r,d

    def get_current_state_info(self):

        pass

    def get_next_state_info(self):
        pass

    def __decode_image(self, image_paths):

        #Node newer images are further back in terms of channel coordinates! 0:3 -> first image .... etc. the last iamge is in the last 3 channels
        image_np = np.concatenate([self.__preprocess_image(misc.imread(image)) for image in image_paths], axis=2)


        return image_np

    def __preprocess_image(self, image):

        if self.mask_path:
            image[self.mask]=0.0

        image = misc.imresize(image, [self.image_size, self.image_size, 3])

        if self.divide_image_values:
            """
            image = image/self.divide_image_values
            image = np.float32(image) #reduce memory usage by 2
            """
            pass


        return image

    def __calculate_next_control_input(self, action, current_state, next_state, current_control_input,fix_step=False):

        # calculate seconds difference between samples
        current_index = current_state.index.values[
            -1]  # Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]

        seconds_diff = (next_index - current_index) / np.timedelta64(1, 's')



        if action == 0:
            ramp_per_sec = 0
        elif action == 1:
            ramp_per_sec = self.max_ramp_per_m / 60
        elif action == 2:
            ramp_per_sec = -self.max_ramp_per_m / 60
        
        elif action == 3:
            ramp_per_sec = (self.max_ramp_per_m / 60)/2
        elif action == 4:
            ramp_per_sec = (-self.max_ramp_per_m / 60)/2
        elif action == 5:
            ramp_per_sec = (self.max_ramp_per_m / 60)/4
        elif action == 6:
            ramp_per_sec = (-self.max_ramp_per_m / 60)/4
        else:
            raise ValueError("Illegal action")

        difference = seconds_diff * ramp_per_sec

        next_control_input = current_control_input + difference

        return next_control_input


    def __calculate_next_control_input_follow_irr(self, action, current_state, next_state, current_control_input,fix_step=False):

        # calculate seconds difference between samples
        current_index = current_state.index.values[
            -1]  # Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]

        seconds_diff = (next_index - current_index) / np.timedelta64(1, 's')

        next_irr = next_state['irr'].values[-1]

        ramp_per_sec = self.max_ramp_per_m / 60

        if action == 0:  # default action, follow current/next irradiance if possible
            step = seconds_diff * ramp_per_sec
            diff = next_irr - current_control_input
            if abs(diff) > step:  # current control input is too far away to get to next irradiance value
                if diff < 0:  # next input goes down
                    step = -step
                next_control_input = current_control_input + step
            else:
                next_control_input = next_irr


        elif action == 1:
            step = seconds_diff * ramp_per_sec
            next_control_input = current_control_input + step

        elif action == 2:
            step = seconds_diff * -ramp_per_sec
            next_control_input = current_control_input + step

        elif action == 3:
            step = seconds_diff * (ramp_per_sec/2)
            next_control_input = current_control_input + step

        elif action == 4:
            step = seconds_diff * (-ramp_per_sec/2)
            next_control_input = current_control_input + step

        elif action == 5:
            step = seconds_diff * (ramp_per_sec / 4)
            next_control_input = current_control_input + step

        elif action == 6:
            step = seconds_diff * (-ramp_per_sec / 4)
            next_control_input = current_control_input + step

        else:
            raise ValueError("Illegal action")

        return next_control_input



    def __calculate_next_control_input_follow_irr_straight0(self, action, current_state, next_state, current_control_input,fix_step=False):
        #Action 0 only follows irradiance if possible, otherwise go straight on, action 0 less powerful
        # calculate seconds difference between samples
        # calculate seconds difference between samples
        current_index = current_state.index.values[
            -1]  # Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]

        seconds_diff = (next_index - current_index) / np.timedelta64(1, 's')

        next_irr = next_state['irr'].values[-1]

        ramp_per_sec = self.max_ramp_per_m / 60

        if action == 0:  # default action, follow current/next irradiance if possible
            step = seconds_diff * ramp_per_sec
            diff = next_irr - current_control_input
            if abs(diff) > step:  # current control input is too far away to get to next irradiance value
                next_control_input = current_control_input # go straight on if cannot reach irradiance
            else:
                next_control_input = next_irr


        elif action == 1:
            step = seconds_diff * ramp_per_sec
            next_control_input = current_control_input + step

        elif action == 2:
            step = seconds_diff * -ramp_per_sec
            next_control_input = current_control_input + step

        elif action == 3:
            step = seconds_diff * (ramp_per_sec / 2)
            next_control_input = current_control_input + step

        elif action == 4:
            step = seconds_diff * (-ramp_per_sec / 2)
            next_control_input = current_control_input + step

        elif action == 5:
            step = seconds_diff * (ramp_per_sec / 4)
            next_control_input = current_control_input + step

        elif action == 6:
            step = seconds_diff * (-ramp_per_sec / 4)
            next_control_input = current_control_input + step

        else:
            raise ValueError("Illegal action")

        return next_control_input




    def __calculate_next_control_input_target_simple(self, action, current_state, next_state, current_control_input,fix_step=False):
        #Defines targets using clear sky model

        current_index = current_state.index.values[
            -1]  # Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]

        seconds_diff = (next_index - current_index) / np.timedelta64(1, 's')

        #next_irr = next_state['irr'].values[-1]
        next_cs = next_state['cs'].values[-1]


        ramp_per_sec = self.max_ramp_per_m / 60
        step = seconds_diff * ramp_per_sec

        d = 0.8/(self.actions-1) #ensure minimum is 0.2

        factor = 1-action*d

        target = next_cs*factor

        diff = target-current_control_input

        if np.abs(diff) > step:

            if diff > 0:
                next_control_input = current_control_input+step
            else:
                next_control_input = current_control_input-step

        else:
            next_control_input = target


        return next_control_input



    def __calculate_step_reward(self, next_irr, next_control_input,action=0):

        if self.reward_type == 1:
            return -np.abs(next_irr - next_control_input)  # reward is negative difference. Maximizing reward is equal to reducing the difference to irr.
        elif self.reward_type == 2:
            return -np.abs(next_irr - next_control_input)/1000
        elif self.reward_type == 3:
            return -np.abs(next_irr - next_control_input)-1.0
        elif self.reward_type == 4:

            if action==0:
                return -np.abs(next_irr - next_control_input) - 20.0 #punish the network's laziness
            else:
                return -np.abs(next_irr - next_control_input) - 1.0

        elif self.reward_type == 5:

            if action==0:
                return -np.abs(next_irr - next_control_input) - 5.0 #punish the network's laziness
            else:
                return -np.abs(next_irr - next_control_input) - 1.0





    def __create_episodes(self, train_set_path, test_set_path):

        print("Environment: Loading rl_data file and datasets...")
        rl_pd = pd.DataFrame.from_csv(self.file_path)

        if train_set_path:
            print("reading " + str(train_set_path))
            with open(str(train_set_path)) as f:
                self.train_list = sorted([os.path.basename(l).split('-', 1)[1] for l in f.read().splitlines()])
        else:
            self.train_list = None

        if test_set_path:
            print("reading " + str(test_set_path))
            with open(str(test_set_path)) as f:
                self.test_list = sorted([os.path.basename(l).split('-', 1)[1] for l in f.read().splitlines()])
        else:
            self.test_list = None

        print("Creating episodes...")
        train_episodes = list()
        test_episodes = list()


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

        if self.load_test_episodes:
            with open(self.load_test_episodes,'rb') as f:
                test_episodes = pickle.load(f)

        else:

            if self.test_list:
                for test_day_it in self.test_list:
                    td_pd = pd.DataFrame(rl_pd.loc[test_day_it])

                    if self.episode_length_test is None:  # 1 day = 1 episode
                        done_pd = np.zeros(len(td_pd.index)).astype(int)
                        done_pd[-1] = 1
                        td_pd["done"] = done_pd
                        test_episodes.append(td_pd)

                    else:

                        for g, episode in td_pd.groupby(np.arange(len(td_pd))//self.episode_length_test):
                            episode_df = pd.DataFrame(episode)
                            done_pd = np.zeros(len(episode_df.index)).astype(int)
                            done_pd[-1] = 1
                            episode_df["done"] = done_pd
                            test_episodes.append(episode_df)


        print("Episodes in Train set:" ,len(train_episodes),"Episodes in Test set:",len(test_episodes))


        if self.sample_training_episodes:
            train_episodes = np.random.choice(train_episodes,size=self.sample_training_episodes)


        return train_episodes, test_episodes



    def mpc_exploration(self, mpc_prob=0.5, num_actions=3):
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

        current_control_input = self.current_episode_train_control_input_values[-1][
            0]  # get last control_input from list
        mpc = np.array(next_state["mpc"].values)[-1]

        control_inputs = list()
        for a in range(num_actions):

            if self.action_space == 1:
                control_inputs.append(
                    abs(self.__calculate_next_control_input_follow_irr(a, current_state, next_state,
                                                                       current_control_input) - mpc))

            elif self.action_space ==2:

                control_inputs.append(
                    abs(self.__calculate_next_control_input_follow_irr_straight0(a, current_state, next_state,
                                                                       current_control_input) - mpc))

            elif self.action_space == 3:

                control_inputs.append(
                    abs(self.__calculate_next_control_input_target_simple(a, current_state, next_state,
                                                                                 current_control_input) - mpc))

            else:
                control_inputs.append(
                    abs(self.__calculate_next_control_input(a, current_state, next_state,
                                                                                 current_control_input) - mpc))




        #best_action = np.argmin(control_inputs[1:])+1 #do not take 0 action into account, only favour non zero
        best_action = np.argmin(control_inputs)        

        action_array = np.arange(0, num_actions, 1)
        normal_action_weight = (1 - mpc_prob) / (num_actions - 1)
        action_weights = np.ones(num_actions) * normal_action_weight
        action_weights[best_action] = mpc_prob

        action = np.random.choice(action_array, replace=False, p=action_weights)

        return action

        # TODO: calculate median step time difference, throw out more outliers (?)
        # TODO: try training on hard samples only ... days with larger errors ...


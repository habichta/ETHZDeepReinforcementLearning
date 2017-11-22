import sys
sys.path.append("/home/dladmin/Documents/arthurma/shared/dlabb")


import numpy as np
import pandas as pd
import random
import os
from scipy import misc
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac
from scipy import ndimage
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_read_pipeline as arp



class Environment():
    def __init__(self,train_set_path,test_set_path,solar_station=ac.ABB_Solarstation.C, image_size=84, sequence_length=2,sequence_stride=6,actions=3,max_ramp_per_m=100,episode_length=None,follow_irr_actions=False):
        self.actions = actions
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.episode_length = episode_length
        self.max_ramp_per_m = max_ramp_per_m
        self.image_size=image_size
        self.follow_irr_actions = follow_irr_actions

        if solar_station==ac.ABB_Solarstation.C:
            self.file_path = os.path.join(ac.c_int_data_path,"rl_data.csv")
            self.img_path = ac.c_img_path
        elif solar_station==ac.ABB_Solarstation.MS:
            self.file_path = os.path.join(ac.ms_int_data_path,"rl_data.csv")
            self.img_path = ac.ms_img_path
        else:
            raise ValueError("Illegal solar station")

        #Episodes:
        self.train_episodes, self.test_episodes =self.__create_episodes(train_set_path=train_set_path, test_set_path=test_set_path)
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

    def reset(self):

        print("Resetting environment...")
        if not self.temp_train_episodes:
            print("Epoch finished...")
            #When all trianing episodes have been sampled at least once, renew the list, start again
            self.temp_train_episodes=list(self.train_episodes)
        print("Sampling episode...")
        #Sample a random episode from the train_episodes list, delete it from list so that it is not sampled in this epoch again
        self.current_episode_train = self.temp_train_episodes.pop(random.randrange(len(self.temp_train_episodes))) #sample episode and remove from temporary list

        print("Episode (from/to): ",str(self.current_episode_train.index[0]),str(self.current_episode_train.index[-1]))

        print("Samples in episode:",len(self.current_episode_train))


        #get index from current eppisode (Datetime)
        index = self.current_episode_train.index.tolist()

        #Create index  for smples depending on image sequence length and stride
        self.train_episode_samples = [index[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride] for i in
                                      range(len(index) - (self.sequence_length - 1) * self.sequence_stride)]

        abort = False
        if (len(self.train_episode_samples) > 1):
            #Set pointer to the current sample, advanced by step()
            self.current_episode_train_step_pointer = 0

            #Get first sample index, list of timestamps of the images and irradiance data
            first_state_index = self.train_episode_samples[self.current_episode_train_step_pointer]

            #Load actual data given the timestamps
            current_state = self.current_episode_train.loc[first_state_index]

            #list of image_names
            images_names = current_state['img_name'].values
            #create paths to images of that sample
            image_paths = [os.path.join(self.img_path,name) for name in images_names]

            #Initialize irradiance and control input
            current_irradiance = np.array(current_state["irr"].values)

            current_control_input = current_irradiance[-1]

            # Reset list that stores all controlinputs for an episode and append first controlinput
            current_timestamp = current_state.index[-1]
            self.current_episode_train_control_input_values=[]
            self.current_episode_train_control_input_values.append((current_control_input,current_timestamp)) #add tuple with control input and timestamp


            #Decode jpeg images and preprocess
            image_tensor = self.__decode_image(image_paths)

            #State:[image: z.b. 84x84x6 tensor, curr_irr float, curr_control_input float]
            first_state = np.array([image_tensor,current_irradiance,current_control_input]) # initial state
        else:
            first_state = None
            abort = True
            print("Episode size is too small, abort this episode")

        return first_state,abort

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

        if self.follow_irr_actions:
            next_control_input = self.__calculate_next_control_input_follow_irr(action, current_state, next_state,
                                                                 current_control_input)
        else:
            next_control_input = self.__calculate_next_control_input(action, current_state, next_state,
                                                                 current_control_input)

        if next_control_input < 0:  # lower boound control input. Otherwise network may train on unrealistic experience of negative control input (there is no negative irradiance)
            next_control_input = 0

        # Update control input list
        next_timestamp = next_state.index[-1]
        self.current_episode_train_control_input_values.append(
            (next_control_input, next_timestamp))  # Add next ocntrol input value

        # reward is negative difference between next irr and next control input. Maximizing reward will reduce difference of irr and control input
        reward = self.__calculate_step_reward(next_irr[-1], next_control_input)

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
            #When all trianing episodes have been sampled at least once, renew the list, start again
            self.temp_test_episodes=list(self.test_episodes)

        #Go along episodes in order
        self.current_episode_test = self.temp_test_episodes.pop() #sample episode and remove from temporary list

        print("Episode (from/to): ", str(self.current_episode_test.index[0]),
              str(self.current_episode_test.index[-1]))
        #get index from current eppisode (Datetime)
        index = self.current_episode_test.index.tolist()

        #Create index  for smples depending on image sequence length and stride
        self.test_episode_samples = [index[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride] for i in
                                      range(len(index) - (self.sequence_length - 1) * self.sequence_stride)]

        abort = False
        if (len(self.test_episode_samples) > 1):  # at least one step should be possible so length must be at least 2
            #Set pointer to the current sample, advanced by step()
            self.current_episode_test_step_pointer = 0

            #Get first sample index, list of timestamps of the images and irradiance data
            first_state_index = self.test_episode_samples[self.current_episode_test_step_pointer]

            #Load actual data given the timestamps
            current_state = self.current_episode_test.loc[first_state_index]

            #list of image_names
            images_names = current_state['img_name'].values
            #create paths to images of that sample
            image_paths = [os.path.join(self.img_path,name) for name in images_names]

            #Initialize irradiance and control input

            current_irradiance = np.array(current_state["irr"].values)
            current_control_input = current_irradiance[-1]

            #Reset list that stores all controlinputs for an episode and append first controlinput
            current_timestamp = current_state.index[-1]
            self.current_episode_test_control_input_values=[]
            self.current_episode_test_control_input_values.append((current_control_input,current_timestamp))

            #Decode jpeg images and preprocess
            image_tensor = self.__decode_image(image_paths)

            #State:[image: z.b. 84x84x6 tensor, curr_irr float, curr_control_input float]
            first_state = np.array([image_tensor,current_irradiance,current_control_input])
        else:
            first_state = None
            abort = True
            print("Episode size is too small, abort this episode")
        return first_state,abort

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

        if self.follow_irr_actions:
            next_control_input = self.__calculate_next_control_input_follow_irr(action, current_state, next_state,
                                                                 current_control_input)
        else:
            next_control_input = self.__calculate_next_control_input(action, current_state, next_state,
                                                                 current_control_input)


        if next_control_input < 0:  # lower boound control input. Otherwise network may train on unrealistic experience of negative control input (there is no negative irradiance)
            next_control_input = 0

        #Update control input list
        next_timestamp = next_state.index[-1]
        self.current_episode_test_control_input_values.append((next_control_input,next_timestamp))

        # reward is negative difference between next irr and next control input. Maximizing reward will reduce difference of irr and control input
        reward = self.__calculate_step_reward(next_irr[-1], next_control_input)

        # done: whether the next state is the last of the episode. Z.b. end of day
        done = next_state.iloc[-1]["done"]

        # Get images of next state
        images_names = next_state['img_name'].values
        image_paths = [os.path.join(self.img_path, name) for name in images_names]
        image_tensor = self.__decode_image(image_paths)


        return np.array([image_tensor,next_irr,next_control_input]),reward,done #return s',r,d


    def get_current_state_info(self):

        pass


    def get_next_state_info(self):
        pass

    def __decode_image(self,image_paths):

        image_np = np.concatenate([self.__preprocess_image(misc.imread(image)) for image in image_paths],axis=2)



        return image_np


    def __preprocess_image(self,image):

        image_prepr =  misc.imresize(image,[self.image_size,self.image_size,3])

        return image_prepr

    def __calculate_next_control_input(self, action, current_state, next_state, current_control_input):

        #calculate seconds difference between samples
        current_index = current_state.index.values[-1] #Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]


        seconds_diff = (next_index-current_index)/np.timedelta64(1, 's')


        if action == 0:
            ramp_per_sec = 0
        elif action == 1:
            ramp_per_sec = self.max_ramp_per_m / 60
        elif action == 2:
            ramp_per_sec = -self.max_ramp_per_m / 60
        else:
            raise ValueError("Illegal action")


        difference = seconds_diff*ramp_per_sec

        next_control_input = current_control_input+difference

        return next_control_input

    def __calculate_next_control_input_follow_irr(self, action, current_state, next_state, current_control_input):

        #calculate seconds difference between samples
        current_index = current_state.index.values[-1]  #Time img1, img2 ,... img n => extract time stamp of last image in state sequence
        next_index = next_state.index.values[-1]

        seconds_diff = (next_index-current_index)/np.timedelta64(1, 's')

        next_irr = next_state['irr'].values[-1]

        ramp_per_sec = self.max_ramp_per_m / 60

        if action == 0: #default action, follow current/next irradiance if possible
            step = seconds_diff * ramp_per_sec
            diff = next_irr - current_control_input
            if abs(diff) >  step : # current control input is too far away to get to next irradiance value
                if diff < 0: # next input goes down
                    step=-step
                next_control_input = current_control_input + step
            else:
                next_control_input = next_irr


        elif action == 1:
            step = seconds_diff * ramp_per_sec
            next_control_input = current_control_input + step

        elif action == 2:
            step = seconds_diff * -ramp_per_sec
            next_control_input = current_control_input + step

        else:
            raise ValueError("Illegal action")

        return next_control_input


    def __calculate_step_reward(self,next_irr,next_control_input):

        return -np.abs(next_irr-next_control_input) # reward is negative difference. Maximizing reward is equal to reducing the difference to irr.

        pass

    def __create_episodes(self,train_set_path,test_set_path):

        print("Environment: Loading rl_data file and datasets...")
        rl_pd = pd.DataFrame.from_csv(self.file_path)



        print("reading "+str(train_set_path))
        with open(str(train_set_path)) as f:
            self.train_list = sorted([os.path.basename(l).split('-', 1)[1] for l in f.read().splitlines()])
        print("reading "+str(test_set_path))
        with open(str(test_set_path)) as f:
            self.test_list = sorted([os.path.basename(l).split('-', 1)[1] for l in f.read().splitlines()])

        print("Creating episodes...")
        train_episodes = list()
        test_episodes = list()

        for train_day_it in self.train_list:
            td_pd = pd.DataFrame(rl_pd.loc[train_day_it])

            if self.episode_length is None:  # 1 day = 1 episode
                done_pd = np.zeros(len(td_pd.index)).astype(int)
                done_pd[-1] = 1
                td_pd["done"] = done_pd
                train_episodes.append(td_pd)
            else:

                for g, episode in td_pd.groupby(np.arange(len(td_pd))//self.episode_length):
                    episode_df = pd.DataFrame(episode)
                    done_pd = np.zeros(len(episode_df.index)).astype(int)
                    done_pd[-1] = 1
                    episode_df["done"] = done_pd
                    train_episodes.append(episode_df)

        for test_day_it in self.test_list:
            td_pd = pd.DataFrame(rl_pd.loc[test_day_it])

            #if self.episode_length is None:  # 1 day = 1 episode
            done_pd = np.zeros(len(td_pd.index)).astype(int)
            done_pd[-1] = 1
            td_pd["done"] = done_pd
            test_episodes.append(td_pd)
            """
            else:

                for g, episode in td_pd.groupby(np.arange(len(td_pd))//self.episode_length):
                    episode_df = pd.DataFrame(episode)
                    done_pd = np.zeros(len(episode_df.index)).astype(int)
                    done_pd[-1] = 1
                    episode_df["done"] = done_pd
                    test_episodes.append(episode_df)
            """
        return train_episodes, test_episodes



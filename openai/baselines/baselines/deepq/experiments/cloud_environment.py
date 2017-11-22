import numpy as np
import cv2
import random
import math
from scipy import misc
import pandas as pd




class CloudEnvironment():




    def __init__(self,img_size=84,radius=[8,16],speed=1,sequence_stride=1,channels=3,sequence_length=2,ramp_step=0.1,action_type=0,action_nr=2,stochastic_irradiance=False,save_images=False):



        self.img_size=img_size
        self.radius = radius
        self.speed = speed
        self.ramp_step=ramp_step
        self.action_type = action_type

        self.sequence_stride=sequence_stride
        self.sequence_length=sequence_length
        self.channels=channels
        self.stochastic_irradiance= stochastic_irradiance
        self.save_images = save_images

        self.observation_space=self.ObservationSpace((img_size*img_size*sequence_length*channels+sequence_length+1,1))

        #self.observation_space

        self.action_space = self.ActionSpace(action_type,action_nr,ramp_step)

        self.episode_samples = []
        self.episode_ci = []
        self.episode_pointer = 0
        self.current_episode = None
        self.episode_nr = 0


    @property
    def current_train_control_inputs(self):
        return self.episode_ci

    @property
    def current_train_episode(self):
        return self.current_episode

    @property
    def episode_n(self):
        return self.episode_nr

    @property
    def episode_id(self):
        return self.episode_nr

    def reset(self):

        self.episode_samples = []
        self.episode_ci = []
        self.episode_pointer = 0

        episode_data_list = self.create_episode() #[(img,irr)]




        self.episode_samples = [episode_data_list[i:(i + (self.sequence_length * self.sequence_stride)):self.sequence_stride]
                                      for i in
                                      range(len(episode_data_list) - (self.sequence_length - 1) * self.sequence_stride)]


        current_episode_step = self.episode_samples[self.episode_pointer]

        images = [img[0] for img in current_episode_step]

        image_tensor = np.uint8(np.concatenate(images,axis=2)) #image tensor with first 3 channels first image, last threee channels newest image
        current_irr = np.array([img[1] for img in current_episode_step])
        current_ci = current_irr[-1]
        self.episode_ci.append(current_ci)

        #env_obs = [np.uint8(image_tensor.ravel()), current_irr, current_ci]
        env_obs = np.concatenate([image_tensor.ravel(), current_irr, np.reshape(current_irr[-1],(1))], axis=0).astype(np.float16)[:, None]

        return env_obs


    def step(self,action):

        curr_pointer = self.episode_pointer
        self.episode_pointer+=1
        next_pointer = self.episode_pointer


        next_episode_step = self.episode_samples[next_pointer]
        curr_episode_step = self.episode_samples[curr_pointer]

        current_ci = self.episode_ci[-1]

        next_irr = np.array([img[1] for img in next_episode_step])
        curr_irr = np.array([img[1] for img in curr_episode_step])

        next_ci,reward = self.action_space.calculate_step(action,next_irr[-1],curr_irr[-1],current_ci.squeeze())

        self.episode_ci.append(next_ci)

        next_images = [img[0] for img in next_episode_step]


        for i in  range(len(next_images)):
            cv2.imshow('Rotation', next_images[i])
            cv2.waitKey(100)
            print(next_irr[-1],self.episode_ci[-1])


        next_image_tensor = np.uint8(np.concatenate(next_images, axis=2))

        if next_pointer == len(self.episode_samples) - 1:
            done = True
        else:
            done = False

        next_env_obs =  np.concatenate([np.uint8(next_image_tensor.ravel()),next_irr, np.reshape(next_ci,(1))]).astype(np.float16)[:, None]
        """
        next_env_obs = [np.uint8(next_image_tensor.ravel()), next_irr, np.reshape(next_ci, (1))]
        cv2.imshow('next_state_image_32', np.float32(np.reshape(next_env_obs[0:-3], (84, 84, 6))[:, :, 0:3]))
        cv2.waitKey(10)
        """


        return next_env_obs, reward , done, 0


    def create_episode(self):

        self.episode_nr += 1
        episode_id = self.episode_n

        episode_data_list = []
        y1 = 0
        x1 = 31#random.randint(0,self.img_size-1)

        y2 = self.img_size-1
        x2 = 61#random.randint(0,self.img_size-1)
        xdiff = x2-x1
        ydiff = y2-y1

        y_update = 1.0 *self.speed
        x_update = (xdiff/ydiff)*self.speed

        center_pixel = self.img_size//2-1

        x_f=float(x1)
        y_f=float(y1)
        x1 = int(x_f)
        y1 = int(y_f)

        #rotation
        rot = 180#np.random.choice([0,90,180,270])
        rotation_matrix = cv2.getRotationMatrix2D((self.img_size/ 2, self.img_size / 2), rot, 1)

        #radius
        radius = 12#np.random.randint(self.radius[0],self.radius[1])


        assert (y1 < self.img_size and x1 >= 0 and x1 < self.img_size)


        while y1 < self.img_size and x1 >= 0 and x1 < self.img_size:
            episode_log = np.array([x1, y1, x2, y2, rot, radius])
            img = np.zeros((self.img_size, self.img_size, self.channels), np.uint8)
            channel_colors = tuple([255]*self.channels)
            cv2.circle(img, (center_pixel, center_pixel),1, tuple([100]*self.channels), -1)
            cv2.circle(img, (x1, y1), radius,channel_colors , -1)
            img=np.uint8(cv2.warpAffine(img, rotation_matrix, (self.img_size, self.img_size)))
            img = np.reshape(img,(self.img_size,self.img_size,self.channels))

            #cv2.imshow('Rotation', img)
            #cv2.waitKey(10)

            if np.abs(x1-center_pixel)<radius and np.abs(y1-center_pixel) <radius:
                irr = 0.0
            else:
                if self.stochastic_irradiance:
                    irr = np.random.uniform(1.0-self.ramp_step,1.0) #still within the ramp to test a follow_irr 0 action
                else:
                    irr = 1.0

            episode_data_list.append((img,irr,episode_id,episode_log))

            x_f += x_update
            y_f += y_update
            x1 = int(x_f)
            y1 = int(y_f)


        assert len(episode_data_list)>0


        col = ['irr', 'id','img_log']
        self.current_episode = pd.DataFrame(data=np.array(episode_data_list)[:,1:], columns=col)

        return episode_data_list

    class ActionSpace():
        def __init__(self, type=0, action_nr=2,ramp_step=0.1):
            self.n = action_nr
            self.type = type
            self.ramp_step = ramp_step

        def calculate_step(self, action, next_irr, curr_irr, current_ci):
            # return next ci given action

            if self.type == 0:

                if action == 0: #upper target
                    next_ci = np.minimum(1.0,float(current_ci)+self.ramp_step)
                else: #lower target
                    next_ci = np.maximum(0.0,current_ci-self.ramp_step)

            elif self.type==-1:
                #naive policy
                diff = next_irr-current_ci
                ramp = np.abs(diff) > self.ramp_step
                if ramp:
                    next_ci = current_ci+np.sign(diff)*self.ramp_step
                else:
                    next_ci = next_irr

            elif self.type==-2:
                #naive policy
                diff = curr_irr-current_ci
                ramp = np.abs(diff) > self.ramp_step
                if ramp:
                    next_ci = current_ci+np.sign(diff)*self.ramp_step
                else:
                    next_ci = curr_irr

            elif self.type==1:

                if action == 0:
                    diff = curr_irr - current_ci
                    ramp = np.abs(diff) > self.ramp_step
                    if ramp:
                        next_ci = current_ci + np.sign(diff) * self.ramp_step
                    else:
                        next_ci = curr_irr

                elif action == 1: #upper target
                    next_ci = np.minimum(1.0,float(current_ci)+self.ramp_step)
                else: #lower target
                    next_ci = np.maximum(0.0,current_ci-self.ramp_step)

            else:

                raise ValueError('Illegal Action Set')

            reward = -np.abs(next_ci - next_irr).squeeze()


            return next_ci,reward


    class ObservationSpace():
        def __init__(self, shape):
            self.shape = shape


"""
env = CloudEnvironment(84,[9,17],2)

d=0
s = env.reset()
while d==0:
    a = random.randint(0,1)

    sn,r,d,_ = env.step(a)

    #print(sn[1][0])
"""






    #Check reward sizes if still ok
    #Testing facility, print predictions ...
    #TODO try adding follow  action in toy environment check what it learns
    #TODO add real evironment and use originam action space to train ..
    #normalize irr in environment in real environment  so  between 0 and 1 this also normalizes reward .. so we have ame situation as in simple environment
    #more varied environment  changing size of cloud random, rotate images randomly



















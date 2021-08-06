
# Import necessary libraries 
import gym
import random 
import time
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np 
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import os


class Agent:



    def __init__(self, env, model_filepath=None):

        self.num_actions = env.action_space.n
        self.model_input_x = 84
        self.model_input_y = 84
        self.env = env


        if model_filepath:

            self.model = load_model(model_filepath)

        else:

            self.model = Sequential([
                Conv2D(32, 8, 4, input_shape=(self.model_input_x, self.model_input_y, 4,),
                    activation='relu'),
                Conv2D(64, 4, 2, activation='relu'),
                Conv2D(64, 3, 1, activation='relu'),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(self.num_actions, activation='linear')
                ]) 

        optimizer = optimizers.RMSprop(learning_rate=0.00025)

        self.model.compile(optimizer=optimizer, loss='huber_loss')

        print(self.model.summary())



    def learn(self, batch, gamma=0.99):

        st = [timestep[0] for timestep in batch]
        at = [timestep[1] for timestep in batch]
        rt = [timestep[2] for timestep in batch]
        st1 = [timestep[3]for timestep in batch]
        done = [timestep[4] for timestep in batch]

        target = self.model.predict([st])

        for t in range(len(batch)):
            if not done[t]:
                st1_q_value = self.model.predict([[st1[t]]])
                target[t, at[t]] = rt[t] + gamma*np.max(st1_q_value)
            else:
                target[t, at[t]] = rt[t]
    
        self.model.fit([st], target, verbose=0)

    


    def act(self, state, epsilon=0.0):

        if np.random.rand() < epsilon:
            action = self.env.action_space.sample()
            
        else:
            q_values = self.model.predict(np.array([state]))
            action = np.argmax(q_values)
            
        return action

         


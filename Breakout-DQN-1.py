#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Import necessary libraries 
import gym
import random 
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np 
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import os


# Initialize the Env
env = make_atari('BreakoutNoFrameskip-v0')
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(42)


class DQN(object):

    
    def __init__(self, env):
        
        self.num_actions = env.action_space.n
        self.model_input_x = 84
        self.model_input_y = 84
        self.env = env
 

        self.model = Sequential([
            Conv2D(32, 8, 4, input_shape=(self.model_input_x, self.model_input_y, 4,),
                activation='relu'),
            Conv2D(64, 4, 2, activation='relu'),
            Conv2D(64, 3, 1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.num_actions, activation='linear')
        ])
        #self.model.load_weights('models/breakout/dqn_300.h5')
        optimizer = optimizers.RMSprop(learning_rate=0.00025, momentum=0.95)
            

        self.model.compile(optimizer=optimizer, loss='huber_loss')

        print(self.model.summary())

    def get_action(self, st, e=0.0):
        '''
        Return the action to take in state st using 
        an e-greedy policy
        '''
        if np.random.rand() < e:
            action = self.env.action_space.sample()
            
        else:
            q_values = self.model.predict(np.array([st]))
            action = np.argmax(q_values)
            
        return action
    
    
        
    def update(self, batch, gamma=0.99):
        
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
                target[t, at[t]] = -5
        
    
        self.model.fit([st], target, verbose=0)
        


def q_learning(dqn, render=False):
    
    
    epsilon = 1
    epsilon_min = 0.1
    eps_interval = epsilon - epsilon_min
    num_random_steps = 20000
    buffer = []
    max_reward = 0
    batch_size = 32
    buffer_size = 50000
    episode_count = 0 
    regular_update = 1
    t = 0 
    episode_rewards = []

    while max_reward <= 40:
    
        episode_reward = 0

        st = env.reset()
            
        while True:

            at = dqn.get_action(st, epsilon)
            st1, rt, done, debug = env.step(at)
            episode_reward += rt

            if render:
                env.render()

            timestep = [np.array(st), at, rt, np.array(st1), done]
            st = st1

            if len(buffer) > buffer_size:
                del buffer[:1]
            buffer.append(timestep)

            
            if t % regular_update == 0:
                batch_size = min(len(buffer), 32)
                batch = random.sample(buffer, batch_size)
                dqn.update(batch)

            
            if epsilon > epsilon_min:
                    #epsilon = epsilon*epsilon_decay
                    epsilon -= eps_interval/num_random_steps

            if epsilon < epsilon_min and episode_count > 1000:
                epsilon_min = 0.001

            if done:      
                episode_rewards.append(episode_reward) 
                print("Episode {}, Reward {}, Epsilon {}".format(episode_count, episode_reward, epsilon))
                if episode_count % 10 == 0:
                    avg_reward = np.average(np.array(episode_rewards[-10:]))
                    print('Average Reward of {} during last Epoch. Epsilon Val {}'.format(avg_reward, epsilon))
                episode_count += 1
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    dqn.model.save('models/breakout/breakout_DQN_{}.h5'.format(episode_reward))
                break

            
            
    
def generate_episode(dqn, render=False):
    '''
    Generates an episode using the dqn
    '''
    episode = []
    
    st = env.reset()

    if render:
        env.render()
    
    while True:
        
        at = dqn.get_action(st)
        
        st1, rt, done, debug = env.step(at)
        
        if render:
            env.render()
        
        timestep = [st, at, rt, st1]
        
        episode.append(timestep)
        
        st = st1
        
        if done:
            break
            
            
    return np.array(episode, dtype=object)


dqn = DQN(env)
q_learning(dqn, render=True)



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
from agent import Agent 

from moviepy.editor import ImageSequenceClip


def train(render=False):

    # Initialize the env and agent 
    env = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    agent = Agent(env)



    epsilon = 1
    epsilon_min = 0.1
    eps_interval = epsilon - epsilon_min
    num_random_steps = 10
    eps_decay = eps_interval/num_random_steps
    buffer = []
    max_reward = 0
    batch_size = 32
    buffer_size = 100000
    episode_count = 0 
    epoch = 0
    num_episodes_per_epoch = 10
    regular_update = 4
    t = 0 
    episode_rewards = []
    lives = 5


    while max_reward <= 400:
        
        episode_reward = 0

        for life in range(lives):

            st = env.reset()
                
            while True:

                at = agent.act(st, epsilon)
                st1, rt, done, debug = env.step(at)
                episode_reward += rt

                if render:
                    env.render()

                timestep = [st, at, rt, st1, done]
                st = st1

                if len(buffer) > buffer_size:
                    del buffer[:1]
                buffer.append(timestep)

                
                if t % regular_update == 0:
                    batch_size = min(len(buffer), 32)
                    batch = random.sample(buffer, batch_size)
                    agent.learn(batch)

 

                t += 1
                
                if epsilon > epsilon_min:
                        #epsilon = epsilon*epsilon_decay
                        epsilon -= eps_decay

                if epsilon < epsilon_min and t > 50000:
                    epsilon_min = 0.000000001
                    eps_decay = (epsilon - epsilon_min) / 1000000


                if done:
                    break

        episode_rewards.append(episode_reward) 
        episode_count += 1 


        print('Episode {} Reward {}'.format(episode_count, episode_reward))
        if episode_count % num_episodes_per_epoch == 0:
            avg_reward = np.average(np.array(episode_rewards[-num_episodes_per_epoch:]))
            print('Epoch {} Average Reward {}'.format(epoch, avg_reward))
            epoch += 1 
            
            
        if episode_reward > max_reward:
            max_reward = episode_reward
            agent.model.save('models/dqn_{}.h5'.format(episode_reward))

        


    
def record(model_filepath='breakout_dqn_97.0.h5', num_episodes=1):

    frames = [] 
    # Initialize the env and agent 
    env = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    agent = Agent(env, model_filepath)

    for i in range(num_episodes):

        print('Recording episode {}...'.format(i))

        st = env.reset()
        frames.append(env.render(mode='rgb_array'))

            
        while True:

            at = agent.act(st)
            st1, rt, done, debug = env.step(at)
            
            frames.append(env.render(mode='rgb_array'))

            st = st1 

            if done:
                break

    
    clip = ImageSequenceClip(frames, fps=40)
    clip.write_gif('sample.gif', fps=40)



    



if __name__ == '__main__':

    record()

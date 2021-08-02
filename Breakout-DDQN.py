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


# Initialize the Env
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, scale=True)
#env.seed(42)


class DQN(object):

    
    def __init__(self, env):
        
        self.num_actions = env.action_space.n
        self.model_input_x = 84
        self.model_input_y = 84
        self.env = env

        model_files = os.listdir('models/breakout/')
        model_files.sort()
        
        
        self.model = load_model('breakout_dqn_97.0.h5')
        self.target_model = load_model('breakout_dqn_97.0.h5')
        
        '''
        self.target_model = Sequential([
        Conv2D(32, 8, 4, input_shape=(self.model_input_x, self.model_input_y, 4,),
            activation='relu'),
        Conv2D(64, 4, 2, activation='relu'),
        Conv2D(64, 3, 1, activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(self.num_actions, activation='linear')

        ])

        self.model = Sequential([
        Conv2D(32, 8, 4, input_shape=(self.model_input_x, self.model_input_y, 4,),
            activation='relu'),
        Conv2D(64, 4, 2, activation='relu'),
        Conv2D(64, 3, 1, activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(self.num_actions, activation='linear')
        ])  
        '''
        optimizer = optimizers.RMSprop(learning_rate=0.00025)
            
        self.model.compile(optimizer=optimizer, loss='huber_loss')
        self.target_model.compile(optimizer=optimizer, loss='huber_loss')

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
                st1_q_value = self.target_model.predict([[st1[t]]])
                target[t, at[t]] = rt[t] + gamma*np.max(st1_q_value)
            else:
                target[t, at[t]] = rt[t]
    
        self.model.fit([st], target, verbose=0)
        


def q_learning(dqn, render=False):
    
    
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
    target_update = 1000
    regular_update = 4
    t = 0 
    episode_rewards = []
    lives = 5

    while max_reward <= 400:
        
        episode_reward = 0

        for life in range(lives):

            st = env.reset()
                
            while True:

                at = dqn.get_action(st, epsilon)
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
                    dqn.update(batch)

                if t % target_update == 0:
                    dqn.target_model.set_weights(dqn.model.get_weights())

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
            dqn.target_model.save('models/breakout_dqn_{}.h5'.format(episode_reward))
            

            
            
    
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
        time.sleep(1/30)
        if render:
            env.render()
        
        timestep = [st, at, rt, st1]
        
        episode.append(timestep)
        
        st = st1
        
        if done:
            break
            
            
    return np.array(episode)


dqn = DQN(env)
q_learning(dqn, render=False)


def demo(num_episodes=5):

    model_files = os.listdir('models/breakout/demo')
    model_files.sort()
    for model_file in model_files:
        model_file_path = 'models/breakout/demo/'+model_file
        print('Loading model {} ...'.format(model_file))
        model = load_model(model_file_path)
        dqn.model = model
        dqn.target_model = model
        for _ in range(num_episodes):
            generate_episode(dqn, render=True)



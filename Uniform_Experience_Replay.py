#class imports
from collections import deque
import numpy as np


class Uniform_Experience_Replay:
    '''
    Uniformly-sampled experience replay buffer
    Stores experiences in the format: pre_state, post_state, action, reward, doneflag
    Args:
        capacity : replay memory buffer length
    '''

    def __init__(self, capacity):
        #initialize deque of size capacity
        self.buffer = deque(maxlen=capacity)
    
    def append(self, experience):
        #append experience stack to deque
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        #random indices of deque
        indices = np.random.default_rng().choice(len(self.buffer), batch_size, replace=False)
        #the zip + * op is going to iterate over all data types in the deque at our indices
        states_1, states_2, actions, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        #return np arrays of unpacked data
        return np.array(states_1), np.array(states_2), np.array(actions), np.array(rewards), np.array(dones) 
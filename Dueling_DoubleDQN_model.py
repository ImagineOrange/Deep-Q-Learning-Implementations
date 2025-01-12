#class imports
import torch
import torch.nn as nn
import numpy as np
import random

class Dueling_DoubleDQN(nn.Module):
    '''
    Dueling Double DQN architecture using Learnable Leaky ReLU activation function.
    Args: 
        input_shape   : [frames stacked x game width x game height] 
        n_actions     : number of possible actions to generate Q-values outputs
        epsilon_start : default beginning epsilon value (model has an epsilon/1 probability of random action, via 
                       epsilon-greedy sampling policy for replay buffer)
        epsilon_min   : epsilon value is annealed over training, until it hits this minimim (>0, or model will loop infinitely)
    '''

    def __init__(self, input_shape, n_actions, epsilon_start, epsilon_min):
        super().__init__()

        #epsilon params
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start

        # nn.Conv2d modules expect a 4-dimensional tensor with the shape [batch_size, channels (frames), height, width]
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),  # Changed to PReLU (learnable alpha), to handle sparsity of input
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.PReLU(),  
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.PReLU(),  
        )

        #advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape), 128),
            nn.PReLU(),  
            nn.Linear(128, n_actions),
        )
        
        #value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape), 128),
            nn.PReLU(),  
            nn.Linear(128, 1),
        )
    

    def get_conv_out(self, input_shape):
         o = self.conv_features(torch.zeros(1, * input_shape))
         o = int(np.prod(o.size()))
         return o


    def forward(self, x):
        #get features from convolutional layer
        conv_out = self.conv_features(x).view(x.size()[0], -1)
        #value from value stream
        value = self.value_stream(conv_out)
        #advantages from advantage stream
        actions_advantages = self.advantage_stream(conv_out)
        #average advantage
        average_advantages  = torch.mean(actions_advantages, dim=1, keepdim=True)

        Q = value + (actions_advantages - average_advantages)

        #return Q values
        return Q

    
    #epsilon greedy sampling policy 
    def uniform_epsilon_greedy(self,frame,observation):
        self.frame = frame
    
        #anneal epsilon over 1.5 million frames
        if self.epsilon > self.epsilon_min:
           self.epsilon = self.epsilon_start - .00000067 * frame
        
        #pick a random number from uniform distribution
        self.rand = random.uniform(0, 1)
        #if epsilon larger, take random action
        if self.rand <= self.epsilon:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action)
        #else, run forward pass on observation
        return self.forward(observation)
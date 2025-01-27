#class imports
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def weight_initialization(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CoefficientNetwork(nn.Module):
    def __init__(self, d_hidden, dropout=0.5):
        super(CoefficientNetwork, self).__init__()
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.layer_norm1 = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.apply(weight_initialization)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        coeff = F.gelu(self.fc1(x))
        coeff = self.layer_norm1(coeff)
        coeff = self.dropout(coeff)
        coeff = torch.sigmoid(self.fc2(coeff))
        return coeff


class DynamicResidualBlock(nn.Module):
    def __init__(self, d_hidden, dropout=0.5):
        super(DynamicResidualBlock, self).__init__()
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.layer_norm1 = nn.LayerNorm(d_hidden)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_hidden)

        self.shortcut = nn.Linear(d_hidden, d_hidden)

        self.coeff_net = CoefficientNetwork(d_hidden)

        self.apply(weight_initialization)


    def forward(self, x):
        coeff = self.coeff_net(x)
        out = self.fc1(x)
        out = self.layer_norm1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        shortcut_out = self.shortcut(x)

        out += coeff * shortcut_out

        return out


class Dynamic_Residual(nn.Module):
    def __init__(self, input_shape, n_actions, epsilon_start, epsilon_min, device, d_hidden=256, dropout=0.5):
        super(Dynamic_Residual, self).__init__()

        self.device = device

        #epsilon params
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start
 
        self.input_dense = nn.Linear(np.prod(input_shape), d_hidden)

        self.advantage_stream1 = DynamicResidualBlock(d_hidden, dropout)
        self.advantage_stream2 = DynamicResidualBlock(d_hidden, dropout)
        self.advantage_stream3 = DynamicResidualBlock(d_hidden, dropout)
        self.advantage_out = nn.Linear(d_hidden, n_actions)

        self.value_stream1 = DynamicResidualBlock(d_hidden, dropout)
        self.value_stream2 = DynamicResidualBlock(d_hidden, dropout)
        self.value_stream3 = DynamicResidualBlock(d_hidden, dropout)
        self.value_out = nn.Linear(d_hidden, 1)

        self.apply(weight_initialization)


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.input_dense(x)

        xA = self.advantage_stream1(x)
        xA = self.advantage_stream2(xA)
        xA = self.advantage_stream3(xA)
        advantage = self.advantage_out(xA).to(self.device)

        xV = self.value_stream1(x)
        xV = self.value_stream2(xV)
        xV = self.value_stream3(xV)
        value = self.value_out(xV).to(self.device)

        average_advantages = torch.mean(advantage, dim=1, keepdim=True).to(self.device)
        Q = value + (advantage - average_advantages)
        
        return Q


    def uniform_epsilon_greedy(self,frame,observation):
        self.frame = frame

        #anneal epsilon over 1.5 million frames
        if self.epsilon > self.epsilon_min:
           self.epsilon = self.epsilon_start - .00000067 * frame
        
        self.rand = random.uniform(0, 1)
        if self.rand <= self.epsilon:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action).to(self.device)

        return self.forward(observation).to(self.device)


    def random_action(self, observation):
        p_random = 5000
        random =  np.random.randint(1, p_random+1)

        if p_random == random:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action).to(self.device)
        else:
            return self.forward(observation).to(self.device)

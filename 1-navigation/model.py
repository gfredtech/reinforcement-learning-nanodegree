import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    '''
    architecture and activation functions of the neural network
    '''

    def __init__(self, state_size, action_size, seed, hidden_size = 64):
        '''
        state_size -> dimension of state
        action_size -> dimension of action
        seed -> fix seed to allow replication
        hidden_size -> number of neurons in each hidden layer
        '''
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        '''
        forward pass of the computation
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
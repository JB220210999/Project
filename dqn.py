import mineflayer
#import mlpack
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQN(nn.module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__() #initialize the parent (ESSENTIAL)
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state)) #math stuff
        x = torch.relu(self.fc2(x)) #math stuff
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size #number of inputs
        self.input_size = action_size #number of outputs
        self.gamma = 0.95 #discount factor
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.01 #minimum exploration rate
        self.epsilon_decay = 0.995 #decay rate of exploration rate
        self.memory = deque(maxlen=2000) #replay buffer
        self.learning_rate = 0.001
        
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) #optimizer for updating weights- could be changed to see how different optimizers perform
        self.criterion = nn.MSELoss() #loss function for tarining

    def memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def act(self, state):
        #chance to explore
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
    
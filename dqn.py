#import mlpack
import torch
import torch.nn as nn

class DQN(nn.module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(DQN, self).__init__() #initialize the parent (ESSENTIAL)
        self.fc1 = nn.Linear(input_size, hidden_size) #input is layer is automatically created so this creats the first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) #second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) #output layer
        
    def forward(self, state):
        first = torch.relu(self.fc1(state)) #math stuff for the first hidden layer
        second = torch.relu(self.fc2(first)) #math stuff for secind hidden layer
        return self.fc3(second)


    
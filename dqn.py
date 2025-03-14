#import mlpack
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, hm_size, vm_size, jump_size, sprint_size, hl_size, vl_size, hidden_size=64):
        super(DQN, self).__init__() #initialize the parent (ESSENTIAL)
        self.fc1 = nn.Linear(input_size, hidden_size) #input is layer is automatically created so this creats the first hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) #second hidden layer
        
        #output layers
        self.horizontal_move = nn.Linear(hidden_size, hm_size) 
        self.vertical_move = nn.Linear(hidden_size, vm_size)
        self.jump = nn.Linear(hidden_size, jump_size)
        self.sprint = nn.Linear(hidden_size, sprint_size)
        self.horizontal_look = nn.Linear(hidden_size, hl_size)
        self.vertical_look = nn.Linear(hidden_size, vl_size)
        
    def forward(self, state):
        first = torch.relu(self.fc1(state)) #math stuff for the first hidden layer
        second = torch.relu(self.fc2(first)) #math stuff for second hidden layer
        
        #output layers
        hm_output = self.horizontal_move(second) 
        vm_output = self.vertical_move(second) 
        jump_output = self.jump(second) 
        sprint_output = self.sprint(second) 
        hl_output = self.horizontal_look(second) 
        vl_output = self.vertical_look(second) 
        return hm_output, vm_output, jump_output, sprint_output, hl_output, vl_output


    
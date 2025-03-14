import random
from dqn import DQN
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class DQNAgent:
    def __init__(self, input_size, hm_size, vm_size, jump_size, sprint_size, hl_size, vl_size):
        #input output size
        self.input_size = input_size 
        self.hm_size = hm_size
        self.vm_size = vm_size
        self.jump_size = jump_size
        self.sprint_size = sprint_size
        self.hl_size = hl_size
        self.vl_size = vl_size
        
        #hyperparameters
        self.discount_factor = 0.95 #gamma
        self.exploration_rate = 0.9 #epsilon
        self.min_exploration = 0.05 #minimum epsilon
        self.exploration_decay = 0.995 #epsilon decay
        self.learning_rate = 0.001
        
        self.memory = deque(maxlen=20000) #replay buffer
        
        #Neural Network model
        self.model = DQN(input_size, hm_size, vm_size, jump_size, sprint_size, hl_size, vl_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) #optimizer for updating weights- could be changed to see how different optimizers perform
        self.criterion = nn.MSELoss() #loss function for tarining


    def remember(self, state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done):
        self.memory.append((state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done)) 


    def choose_action(self, state):
        #chance to explore
        if np.random.rand() <= self.exploration_rate: 
            return np.random.randint(self.hm_size), np.random.randint(self.vm_size), np.random.randint(self.jump_size), np.random.randint(self.sprint_size), np.random.randint(self.hl_size), np.random.randint(self.vl_size)
        
        #create tensor and make it the correct shape
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        #get q values
        with torch.no_grad():
            hm_q, vm_q, jump_q, sprint_q, hl_q, vl_q = self.model(state_tensor)
            
            hm_action = np.argmax(hm_q).item()
            vm_action = np.argmax(vm_q).item()
            jump_action = np.argmax(jump_q).item()
            sprint_action = np.argmax(sprint_q).item()
            hl_action = np.argmax(hl_q).item()
            vl_action = np.argmax(vl_q).item()
        return hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action 
        
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        sample = random.sample(self.memory, batch_size)
        
        for state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done in sample:
            #create tensors and make them the correct shapes for the current and next states
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            hm_target, vm_target, jump_target, sprint_target, hl_target, vl_target  = self.model(state_tensor)#get the q values for the current state and detach from the graph
            #hm_target, vm_target, jump_target, sprint_target, hl_target, vl_target = hm_target.detach(), vm_target.detach(), jump_target.detach(), sprint_target.detach(), hl_target.detach(), vl_target.detach()
            
            if done:
                hm_target[0][hm_action] = reward
                vm_target[0][vm_action] = reward
                jump_target[0][jump_action] = reward
                sprint_target[0][sprint_action] = reward
                hl_target[0][hl_action] = reward
                vl_target[0][vl_action] = reward
            else:
                with torch.no_grad():
                    hm_next_q, vm_next_q, jump_next_q, sprint_next_q, hl_next_q, vl_next_q = self.model(next_state_tensor) #get the q values for the next state
                    
                    #adding the future max q value multiplied by discount factor to the original q value
                    hm_target[0][hm_action] = reward + self.discount_factor * torch.max(hm_next_q) 
                    vm_target[0][vm_action] = reward + self.discount_factor * torch.max(vm_next_q)
                    jump_target[0][jump_action] = reward + self.discount_factor * torch.max(jump_next_q)
                    sprint_target[0][sprint_action] = reward + self.discount_factor * torch.max(sprint_next_q)
                    hl_target[0][hl_action] = reward + self.discount_factor * torch.max(hl_next_q)
                    vl_target[0][vl_action] = reward + self.discount_factor * torch.max(vl_next_q)
                    
            self.optimizer.zero_grad() #resets the gradients
            hm_predicted, vm_predicted, jump_predicted, sprint_predicted, hl_predicted, vl_predicted = self.model(state_tensor) #predicts q values for the current state
            
            #measures difference between predictions and targets then summing them
            hm_loss = self.criterion(hm_predicted, hm_target)
            vm_loss = self.criterion(vm_predicted, vm_target)
            jump_loss = self.criterion(jump_predicted, jump_target)
            sprint_loss = self.criterion(sprint_predicted, sprint_target)
            hl_loss = self.criterion(hl_predicted, hl_target)
            vl_loss = self.criterion(vl_predicted, vl_target)
            loss = hm_loss + vm_loss + jump_loss + sprint_loss + hl_loss + vl_loss 
            loss.backward() #backpropogates weird graph gradient math
            self.optimizer.step() #updates weights based on weird graph gradient math
            
        if self.exploration_rate > self.min_exploration: 
            self.exploration_rate = max(self.min_exploration, self.exploration_rate - 0.0005)
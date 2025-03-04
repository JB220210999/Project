import random
from dqn import DQN
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mineflayer


class DQNAgent:
    def __init__(self, state_size, action_size):
        #input output size
        self.state_size = state_size 
        self.input_size = action_size 
        
        #hyperparameters
        self.discount_factor = 0.95 #gamma
        self.exploration_rate = 1.0 #epsilon
        self.min_exploration = 0.01 #minimum epsilon
        self.exploration_decay = 0.995 #epsilon decay
        self.learning_rate = 0.001
        
        #replay buffer
        self.memory = deque(maxlen=2000) 
        
        #Neural Network
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) #optimizer for updating weights- could be changed to see how different optimizers perform
        self.criterion = nn.MSELoss() #loss function for tarining

    def memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def choose_action(self, state):
        #chance to explore
        if np.random.rand() <= self.exploration_rate: 
            return np.random.randint(self.action_size)
        
        #create tensor and make it the correct shape
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        #get q values
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values).item()
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        sample = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in sample:
            #create tensors and make them the correct shapes for the current and next states
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            target = self.model(state_tensor).detach() #get the q values for the current state and detach from the graph
            
            if done:
                target[0][action] = reward
            else:
                with torch.no_grad():
                    target[0][action] = reward + self.discount_factor * torch.max(self.model(next_state_tensor)) #adds the future max q value multiplied by discount factor to the original q value
            self.optimizer.zero_grad() #resets the gradients
            predicted = self.model(state_tensor) #predicts q values for the current state
            loss = self.criterion(predicted, target) #measures difference between prediction and target
            loss.backward() #backpropogates weird graph gradient math
            self.optimizer.step() #updates weights based on weird graph gradient math
        if self.exploration_rate > self.min_exploration: 
            self.exploration_rate *= self.exploration_decay
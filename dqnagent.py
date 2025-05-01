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
        self.discount_factor = 0.98 #gamma
        self.exploration_rate = 0.8 #epsilon
        self.min_exploration = 0.08 #minimum epsilon
        self.exploration_decay = 0.999 #epsilon decay
        self.learning_rate = 0.001
        
        self.memory = deque(maxlen=20000) #replay buffer
        
        #Neural Network model
        self.model = DQN(input_size, hm_size, vm_size, jump_size, sprint_size, hl_size, vl_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) #optimizer for updating weights- could be changed to see how different optimizers perform
        self.criterion = nn.MSELoss() #loss function for tarining


    def remember(self, state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done):
        self.memory.append((state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done)) 


    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        #get q values
        with torch.no_grad():
            hm_q, vm_q, jump_q, sprint_q, hl_q, vl_q = self.model(state_tensor)
        
        #chance to explore
        if np.random.rand() <= self.exploration_rate: 
            return (
            np.random.randint(self.hm_size),
            np.random.randint(self.vm_size),
            np.random.randint(self.jump_size),
            np.random.randint(self.sprint_size),
            np.random.randint(self.hl_size),
            np.random.randint(self.vl_size),
        )
        
        def softmax_sample(q_values):
            q_values = q_values.cpu().numpy().flatten()  # Convert tensor to NumPy
            probs = torch.nn.functional.softmax(torch.tensor(q_values), dim=0).numpy()
            return np.random.choice(len(q_values), p=probs)
        
        return (
            softmax_sample(hm_q),
            softmax_sample(vm_q),
            softmax_sample(jump_q),
            softmax_sample(sprint_q),
            softmax_sample(hl_q),
            softmax_sample(vl_q),
        )
        
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        sample = random.sample(self.memory, batch_size)
        batch = list(zip(*sample))  

        #converting tensors
        states = torch.FloatTensor(batch[0])  #state tensor
        rewards = torch.FloatTensor(batch[7])  #reward tensor
        next_states = torch.FloatTensor(batch[8])  #next state tensor
        dones = torch.BoolTensor(batch[9])  #done tensor

        #actions must be separated per action dimension making it way harder but ITS FINE
        hm_actions = torch.LongTensor(batch[1])
        vm_actions = torch.LongTensor(batch[2])
        jump_actions = torch.LongTensor(batch[3])
        sprint_actions = torch.LongTensor(batch[4])
        hl_actions = torch.LongTensor(batch[5])
        vl_actions = torch.LongTensor(batch[6])

        #get current Q-values from model
        hm_q, vm_q, jump_q, sprint_q, hl_q, vl_q = self.model(states)

    #get next state Q-values
        with torch.no_grad():
            hm_next_q, vm_next_q, jump_next_q, sprint_next_q, hl_next_q, vl_next_q = self.model(next_states)

    #clone target Q-values to update and detach from the graph to avoid problems
        hm_target, vm_target, jump_target, sprint_target, hl_target, vl_target = (
            hm_q.clone().detach(),
            vm_q.clone().detach(),
            jump_q.clone().detach(),
            sprint_q.clone().detach(),
            hl_q.clone().detach(),
            vl_q.clone().detach(),
        )

        #update target Q-values using the Bellman equation
        for i in range(batch_size):
            if dones[i]:  
                hm_target[i, hm_actions[i]] = rewards[i]
                vm_target[i, vm_actions[i]] = rewards[i]
                jump_target[i, jump_actions[i]] = rewards[i]
                sprint_target[i, sprint_actions[i]] = rewards[i]
                hl_target[i, hl_actions[i]] = rewards[i]
                vl_target[i, vl_actions[i]] = rewards[i]
            else:  
                hm_target[i, hm_actions[i]] = rewards[i] + self.discount_factor * torch.max(hm_next_q[i])
                vm_target[i, vm_actions[i]] = rewards[i] + self.discount_factor * torch.max(vm_next_q[i])
                jump_target[i, jump_actions[i]] = rewards[i] + self.discount_factor * torch.max(jump_next_q[i])
                sprint_target[i, sprint_actions[i]] = rewards[i] + self.discount_factor * torch.max(sprint_next_q[i])
                hl_target[i, hl_actions[i]] = rewards[i] + self.discount_factor * torch.max(hl_next_q[i])
                vl_target[i, vl_actions[i]] = rewards[i] + self.discount_factor * torch.max(vl_next_q[i])

        self.optimizer.zero_grad()
        loss = (self.criterion(hm_q, hm_target)+ self.criterion(vm_q, vm_target)+ self.criterion(jump_q, jump_target)+ self.criterion(sprint_q, sprint_target)+ self.criterion(hl_q, hl_target)+ self.criterion(vl_q, vl_target))
    
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.min_exploration, self.exploration_rate)

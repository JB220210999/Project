# Import python libraries required in this example:
import random
import numpy as np
from collections import deque
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
# DEFINE THE NETWORK
# Generate random numbers within a truncated (bounded) 
# normal distribution:
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
 # Create the ‘Nnetwork’ class and define its arguments:
# Set the number of neurons/nodes for each layer
# and initialize the weight matrices:  
class Nnetwork:
    def __init__(self, input_size, output_size, hidden_size=16, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.input_size)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.hidden_size, self.input_size))
        rad = 1 / np.sqrt(self.hidden_size)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.output_size, self.hidden_size))

    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_size)  # Explore
        q_values = self.run(state)
        return np.argmax(q_values)  # Exploit
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        """Train the network using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.run(state)
            if done:
                target[action] = reward  # Terminal state
            else:
                next_q_values = self.run(next_state)
                target[action] = reward + self.gamma * np.max(next_q_values)  # Bellman Equation
            
            self.train(state, target)

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, input_vector, target_vector):
        """Update weights using gradient descent"""
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector = activation_function(self.weights_hidden_out @ input_hidden)

        error = target_vector - output_vector
        delta_out = error * output_vector * (1 - output_vector)
        delta_hidden = (self.weights_hidden_out.T @ delta_out) * input_hidden * (1 - input_hidden)

        # Update weights
        self.weights_hidden_out += self.learning_rate * (delta_out @ input_hidden.T)
        self.weights_in_hidden += self.learning_rate * (delta_hidden @ input_vector.T)
            
    def run(self, input_vector):
        """Forward pass to compute Q-values"""
        input_vector = np.array(input_vector, ndmin=2).T
        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector = activation_function(self.weights_hidden_out @ input_hidden)
        return output_vector.flatten()
 # RUN THE NETWORK AND GET A RESULT
# Initialize an instance of the class:  
simple_network = Nnetwork(input_size=2, 
                               output_size=2, 
                               hidden_size=4,
                               learning_rate=0.6)
 # Run simple_network for arrays, lists and tuples with shape (2):
# and get a result:
simple_network.run([(3, 4)])
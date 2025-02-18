import numpy as np


class NeuralNetwork():
    def __init__(self):
        
        np.random.seed(1)
        
        self.weight_matrix = 2 * np.random.random((3,1)) - 1
        
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, inputs, outputs, training_iterations):
        
        for iteration in range(training_iterations):
            
            output = self.think(inputs)
            error = outputs - output
            adjustment = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            
            self.weight_matrix += adjustment
    
    def think(self, inputs):
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.weight_matrix))
        return output
    
if __name__ == "__main__":
    
    neural_network = NeuralNetwork()
    
    print("Random weights at the start of training")
    print(neural_network.weight_matrix)
    
    training_inputs = np.array([[0,0,1],
                                 [1,1,1],
                                 [1,0,1],
                                 [0,1,1]])
    
    training_outputs = np.array([[0,1,1,0]]).T
    
    neural_network.train(training_inputs, training_outputs, 10000)
    
    print("Synaptic weights after training")
    print(neural_network.weight_matrix)
    
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))

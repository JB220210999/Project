import numpy as np


class NeuralNetwork():
    def __init__(self):
        
        #Set seed so all random values are the same at start
        np.random.seed(1)
        
        #Create 3x1 matrix for weights
        self.weight_matrix = 2 * np.random.random((3,1)) - 1
        
    #Sigmoid function as activation function- CHANGE TO RELU
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    #Takes inputs outputs and iterations- REMOVE ITERATION LIMIT, CHANEG TO COMPLETING THE COURSE
    def train(self, inputs, outputs, training_iterations):
        
        for iteration in range(training_iterations):
            
            output = self.think(inputs)
            error = outputs - output
            #This is used for backpropagation which we will not need- REPLACE WITH Q LEARNING UPDATE RULE
            adjustment = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            
            self.weight_matrix += adjustment
    
    def think(self, inputs):
        
        inputs = inputs.astype(float)
        #Applies the sigmoid function to the dot product to create activation value
        output = self.sigmoid(np.dot(inputs, self.weight_matrix))
        return output
    
if __name__ == "__main__":
    
    neural_network = NeuralNetwork()
    
    print("Synaptic weights before training")
    print(neural_network.weight_matrix)
    
    #Training inputs- CHANGE TO INPUTS FROM GAME E.G POSITION OF BOT, POSITION OF NEXT JUMP
    training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    
    #Training outputs- CHANGE TO OUTPUTS FROM GAME E.G MOVE FORWARD, JUMP
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

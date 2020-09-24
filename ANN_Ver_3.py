# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:07:26 2020

@author: filip
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    net_input = [0.05, 0.1]
   
    n_node_arr = [2,2,2] # Array determining size of NN. len of array correspond to nr of layers and numbers correspond to nodes in each layer
    learning_rate = 0.1
    n_laps = 5000
    nn = NeuralNetwork(n_node_arr, learning_rate)
    
    rmsError = nnBootCamp(nn, n_laps, net_input)
    
    print('Plotting average error:')
    plt.plot(rmsError)
    plt.axis([0, n_laps, 0, 1])
    
    #nnWorkCamp(nn, net_input)
    
    
#def nnWorkCamp(nn, net_input):
#    for i in range(10):
#        net_output = nn.feedforward(net_input)
#        
#       classified_letter = alphabet[np.argmax(net_output[-1])]
#       # print(net_output[-1])
#       print('Classified nr: ' + str(classified_letter) + '| Correct nr: ' + str(alphabet[i]))
    

def nnBootCamp(nn, n_laps, net_input):
    aveError = []
    targets = [0.01,0.99]
    for i in range(n_laps):
        #targets = [0.0] * len(alphabet)
        #letter = alphabet[random.randint(0, len(alphabet) - 1)]
        #targets[alphabet.index(letter)] = 1.0
        
        aveError.append(getAverageError(nn, net_input, targets))
        
        nn.backpropagation(net_input, targets)
    return aveError


def getAverageError(nn, net_input, targets):
    error = 0
    net_output = nn.feedforward(net_input)
    for i in range(len(net_output[-1])):
        error += (net_output[-1][i] - targets[i]) ** 2
    return math.sqrt(error) / len(net_output[-1])
    

def getInData(alphabet):
    net_input = {}
    
    for letter in alphabet:
        temp = []
        file = open('TrainingDataA-Z/' + letter + '.csv', "r")
        file_data = file.readlines()
        for line in file_data:
            bin_arr = line.split(",")
            for elem in bin_arr:
                temp.append(int(elem.strip('\n')))
        net_input[letter] = temp
    
    return net_input # Dictionary with letter as key and flattened matrix as value.
    

class NeuralNetwork:
    def __init__(self, n_node_arr, learning_rate):
        self.n_layers = len(n_node_arr) # n_nodes[0] = input, [-1] = output. All other hidden.
        self.lr = learning_rate
        self.weight_matrix_array = []
        self.bias_array_array = []
        self.map_sigmoid = np.vectorize(self.sigmoid)
        self.map_dsigmoid = np.vectorize(self.dsigmoid)
        
        # Generate weight matrixes:
        for i in range(0,self.n_layers - 1):
            self.weight_matrix_array.append(np.random.rand(n_node_arr[i + 1], n_node_arr[i]))
            self.bias_array_array.append(np.random.rand(n_node_arr[i + 1]))
        
        
        ################## Changing to Matt Mazur's values: ###################
        w1 = [[0.15,0.25],[0.2,0.3]]
        w2 = [[0.4,0.5],[0.45,0.55]]
        
        b1 = np.array([0.35,0.35])
        b2 = np.array([0.60,0.60])
        
        for i in range(len(self.weight_matrix_array[0])):
            for j in range(len(self.weight_matrix_array[0][i])):
                self.weight_matrix_array[0][i][j] = w1[i][j]
                self.weight_matrix_array[1][i,j] = w2[i][j]
        for i in range(len(self.bias_array_array[0])):
            self.bias_array_array[0] = b1[i]
            self.bias_array_array[1] = b2[i]
        #######################################################################
        
        
    def feedforward(self, input_arr):
        inputs = np.array(input_arr)
        node_values = []
        node_values.append(inputs)
        #print(node_values[-1].shape)
        for i in range(self.n_layers - 1):
            node_values.append(self.map_sigmoid(self.weight_matrix_array[i] @ node_values[i] + self.bias_array_array[i]))
        return node_values
    
    
    def backpropagation(self, input_arr, targets):
        errors = []
        z = []      # Storage variable, z = w*a + b, a = node_values (result after activation function in each node)
        dC_dw = []  # derivative of Cost resp to weights. dC_dw = da_dz*dC_da*dz_dw
        dC_db = []  # derivative of Cost resp to bias. dC_db = dz_db*da_dz*dC_da
        
        dz_dw = []  # derivative of z WRT weights
        da_dz = []  # derivative of node values after activation function WRT z
        dC_da = []  # derivative of cost function WRT node values
        dz_db = 1   # Always 1.
        
        node_values = self.feedforward(input_arr)
        targets = np.array(targets)
        output_errors = node_values[-1] - targets
        
        rmsError = self.getRmsError(output_errors)
        
        for i in range(0, self.n_layers - 1):
            errors.append(self.propagate_error(i, output_errors))
            z.append(self.weight_matrix_array[-(i+1)] @ node_values[-(i+2)] + self.bias_array_array[-(i+1)])
    
            dz_dw.append(node_values[-(i+2)]) # Sista varvet ska vara inputs här.
            da_dz.append(self.map_dsigmoid(z[-1]))
            
            dC_da.append(2*errors[-1])
            # dC_da(L-1) = Sum(dz_da(L-1)*da_dz(L)*dC_da(L))
            
            dC_dw.append(da_dz[-1][:,None] * dC_da[-1][:,None] @ np.transpose(dz_dw[-1][:,None]))
            dC_db.append(dz_db * da_dz[-1] * dC_da[-1])
            
            self.weight_matrix_array[-(i+1)] -= self.lr*dC_dw[-1]
            self.bias_array_array[-(i+1)] -= self.lr*dC_db[-1]

        return rmsError

    
    def propagate_error(self, i, output_errors):
        if (i == 0):
            return output_errors
        else:
            return np.transpose(self.weight_matrix_array[-i]) @ self.propagate_error(i-1, output_errors)
    
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def getRmsError(self, output_errors):
        error = 0
        for i in range(len(output_errors)):
            error += output_errors[i] ** 2
            
        return math.sqrt(error / len(output_errors))
    
main()

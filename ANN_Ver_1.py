# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:07:26 2020

@author: filip
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def main():
    alphabet = ['A','B','C']#,'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    net_input = getInData(alphabet)
   
    n_node_arr = [len(net_input['A']),5, len(net_input.keys())] # Array determining size of NN. len of array correspond to nr of layers and numbers correspond to nodes in each layer
    learning_rate = 0.01
    n_laps = 5000
    nn = NeuralNetwork(n_node_arr, learning_rate)
    
    rmsError = nnBootCamp(nn, n_laps, alphabet, net_input)
    
    plt.plot(rmsError)
    plt.axis([0, n_laps, 0, 1])
    
    print('Classifying in progress!')
    nnWorkCamp(nn, net_input, alphabet)
    
    
def nnWorkCamp(nn, net_input, alphabet):
    for i in range(len(alphabet)):
        net_output = nn.feedforward(net_input[alphabet[i]])
        print(net_output[-1])
        classified_letter = alphabet[np.argmax(net_output[-1])]
        # print(net_output[-1])
        print('Classified nr: ' + str(classified_letter) + '| Correct nr: ' + str(alphabet[i]))
    

def nnBootCamp(nn, n_laps, alphabet, net_input):
    aveError = []
    for i in range(n_laps):
        targets = [0.0] * len(alphabet)
        letter = alphabet[random.randint(0, len(alphabet) - 1)]
        targets[alphabet.index(letter)] = 1.0
        
        aveError.append(getAverageError(nn, net_input[letter], targets))
        
        nn.backpropagation(net_input[letter], targets)
    return aveError

def getAverageError(nn, net_input, targets):
    error = 0
    net_output = nn.feedforward(net_input)
    for i in range(len(net_output[-1])):
        error += (net_output[-1][i] - targets[i]) ** 2
    return math.sqrt(error / len(net_output[-1]))
    

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
            #print(self.weight_matrix_array[-1].shape)
            #self.bias_array_array.append(np.random.rand(n_node_arr[i + 1]))
            self.bias_array_array.append(np.zeros(n_node_arr[i + 1]))
            #print(self.bias_array_array[-1].shape)
        
    def feedforward(self, input_arr):
        inputs = np.array(input_arr)
        node_values = []
        node_values.append(inputs)
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
            # dC_da
            
            dC_dw.append(da_dz[-1][:,None] * dC_da[-1][:,None] @ np.transpose(dz_dw[-1][:,None]))
            dC_db.append(dz_db * da_dz[-1] * dC_da[-1])
            
            self.weight_matrix_array[-(i+1)] -= self.lr*dC_dw[-1]*np.linalg.norm(dC_dw[-1])
            self.bias_array_array[-(i+1)] -= self.lr*dC_db[-1]*np.linalg.norm(dC_db[-1])
            
            
        return rmsError, np.linalg.norm(dC_dw[-1])

    
    def propagate_error(self, i, output_errors):
        if (i == 0):
            return output_errors
        else:
            return np.transpose(self.weight_matrix_array[-i]) @ self.propagate_error(i-1, output_errors)
            # return  * self.propagate_error(i-1, da_dz, output_errors)
    
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

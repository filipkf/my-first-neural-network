# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:07:26 2020

@author: filip
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import idx2numpy
import timeit

def main():

    convoluteTrainingData = False # Set to True if convolution is required
    # in_data_init_dim = 28       # Initial input data dimension (assuming square matrix (n by n))
    n_conv_laps = 2             # Nr of convolutional laps
    if convoluteTrainingData:
        training_data, training_labels, validating_data, validating_labels = getConvolutedData(n_conv_laps)
    else:
        training_data, training_labels, validating_data, validating_labels = getData()

    n_node_arr = [784, 16, 16, 10] # Nr of elements correspond to nr of layers, and the value is the nr of nodes in each layer.
    learning_rate = 0.001
    n_laps = 1
    nn = NeuralNetwork(n_node_arr, learning_rate)
    
    rmsError = nnBootCamp(nn, n_laps, training_labels, training_data)
    
    plt.plot(rmsError)
    plt.axis([0, n_laps, 0, 1])
    
    print('Classifying in progress!')
    nnWorkCamp(nn, validating_data, validating_labels)

    
def nnWorkCamp(nn, validating_data, validating_labels):
    for i in range(10): #len(validating_labels)
        net_output = nn.feedforward(validating_data[i])
        
        classified_nr = np.argmax(net_output[-1])
        # print(net_output[-1])
        print('Classified nr: ' + str(classified_nr) + '| Correct nr: ' + str(validating_labels[i]))
        print(net_output[-1])
        
    
    
def nnBootCamp(nn, n_laps, labels, training_data):
    rmsError = []
    for i in range(n_laps):
        rand_sample = random.randint(0, len(labels) - 1)
        rmsError.append(nn.backpropagation(training_data[rand_sample], labels[rand_sample]))
        if i%(n_laps/10) == 0:
            print('Training in progress: ' + str(int(100*i/n_laps)) + '% complete!')
    print('Training completed!\n')
    return rmsError

        
class NeuralNetwork:
    def __init__(self, n_node_arr, learning_rate):
        self.n_layers = len(n_node_arr) # n_nodes[0] = input, [-1] = output. All other hidden.
        self.lr = learning_rate
        self.weight_matrix_array = []
        self.bias_array_array = []
        self.map_sigmoid = np.vectorize(self.sigmoid)
        self.map_dsigmoid = np.vectorize(self.dsigmoid)
        # Generate randomized weight and bias matrixes:
        for i in range(0,self.n_layers - 1):
            self.weight_matrix_array.append(np.random.rand(n_node_arr[i + 1], n_node_arr[i]))
            self.bias_array_array.append(np.random.rand(n_node_arr[i + 1]))
     
        
    def feedforward(self, input_arr):
        inputs = self.map_sigmoid(np.array(input_arr))
        node_values = []
        node_values.append(inputs)
        for i in range(self.n_layers - 1):
            #if (i == 0):
            #    node_values.append(self.map_sigmoid(self.weight_matrix_array[i] @ inputs + self.bias_array_array[i]))
                # print(str(self.weight_matrix_array[i].shape) + ' times ' + str(inputs.shape) + ' + ' + str(self.bias_array_array[i].shape) + 'became a ' + str(node_values[i].shape))
            #else:
            node_values.append(self.map_sigmoid(self.weight_matrix_array[i] @ node_values[i] + self.bias_array_array[i]))
                # print(str(self.weight_matrix_array[i].shape) + ' times ' + str(values[i - 1].shape) + ' + ' + str(self.bias_array_array[i].shape) + 'became a ' + str(values[i].shape))
        return node_values
    
    
    def backpropagation(self, input_arr, targets):
        errors = []
        z = []      # Storage variable, z = w*a + b, a = node_values (result after activation function in each node in the layer before)
        dC_dw = []  # derivative of Cost w.r.t. weights. dC_dw = da_dz*dC_da*dz_dw
        dC_db = []  # derivative of Cost w.r.t. bias. dC_db = dz_db*da_dz*dC_da
        
        dz_dw = []  # derivative of z w.r.t. weights
        da_dz = []  # derivative of node values after activation function w.r.t. z
        dC_da = []  # derivative of cost function w.r.t. node values
        # dz_db = 1 # Always 1.
        
        node_values = self.feedforward(input_arr)
        targets = np.array(targets)
        output_errors = node_values[-1] - targets
        
        rmsError = self.getRmsError(output_errors)
        
        for i in range(0, self.n_layers - 1):
            errors.append(self.propagate_error(i, output_errors))
            z.append(self.weight_matrix_array[-(i+1)] @ node_values[-(i+2)] + self.bias_array_array[-(i+1)])
    
            dz_dw.append(node_values[-(i+2)]) # a(L-1), värdet av aktiveringsfunktion i lager L-1.
            da_dz.append(self.map_dsigmoid(z[-1])) #a'(L), värdet av derivatan av aktiveringsfunktionen i lager L
            dC_da.append(2*errors[-1])
            
            dC_dw.append(da_dz[-1][:,None] * dC_da[-1][:,None] @ np.transpose(dz_dw[-1][:,None]))
            dC_db.append(da_dz[-1] * dC_da[-1])
            
            self.weight_matrix_array[-(i+1)] -= self.lr*dC_dw[-1]
            self.bias_array_array[-(i+1)] -= self.lr*dC_db[-1]
            
        return rmsError
   
    
    def propagate_error(self, i, output_errors):
        if (i == 0):
            return output_errors
        else:
            return np.transpose(self.weight_matrix_array[-i]) @ self.propagate_error(i-1, output_errors)
 
    
    def getRmsError(self, output_errors):
        error = 0
        for i in range(len(output_errors)):
            error += output_errors[i] ** 2
            
        return math.sqrt(error / len(output_errors))


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))    





def getData():
    print('Opening MNIST data base...')
    map_divide_by_255 = np.vectorize(divide_by_255)
    
    train_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_training_data = idx2numpy.convert_from_file(train_data_file)
    raw_training_data = map_divide_by_255(raw_training_data)
    
    train_label_file = open('train-labels.idx1-ubyte', 'rb')
    raw_training_labels = idx2numpy.convert_from_file(train_label_file)
    
    validating_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_validating_data = idx2numpy.convert_from_file(validating_data_file)
    raw_validating_data = map_divide_by_255(raw_validating_data)
    
    validating_label_file = open('train-labels.idx1-ubyte', 'rb')
    raw_validating_labels = idx2numpy.convert_from_file(validating_label_file)  
    print('Data sucessfully read into memory!')
    print('\nFlattening data...')
    
    flattened_training_data = []
    training_label_arrays = []

    for i in range(len(raw_training_data)):
        flattened_training_data.append(raw_training_data[i].flatten())
        label_as_array = [0.0] * 10
        label_as_array[raw_training_labels[i]] = 1.0
        training_label_arrays.append(label_as_array)
    
    
    flattened_validating_data = []
    #validating_label_arrays = []
    
    for i in range(len(raw_validating_data)):
        flattened_validating_data.append(raw_validating_data[i].flatten())
        #label_as_array = [0.0] * 10
        #label_as_array[raw_validating_labels[i] - 1] = 1.0
        #validating_label_arrays.append(label_as_array)
        
    print('Data sucessfully flattened!\n')
    
    return flattened_training_data, training_label_arrays, flattened_validating_data, raw_validating_labels#validating_label_arrays


def divide_by_255(x):
    return x/255


def getConvolutedData(n_conv_laps):
    # move to separate method
    print('Opening MNIST data base...')
    map_divide_by_255 = np.vectorize(divide_by_255)
    
    train_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_training_data = idx2numpy.convert_from_file(train_data_file)
    raw_training_data = map_divide_by_255(raw_training_data)
    
    train_label_file = open('train-labels.idx1-ubyte', 'rb')
    raw_training_labels = idx2numpy.convert_from_file(train_label_file)
    
    validating_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_validating_data = idx2numpy.convert_from_file(validating_data_file)
    raw_validating_data = map_divide_by_255(raw_validating_data)
    
    validating_label_file = open('train-labels.idx1-ubyte', 'rb')
    raw_validating_labels = idx2numpy.convert_from_file(validating_label_file)  
    print('Data sucessfully read into memory!')
    print('\nConvoluting data...')

    vert_filt = np.matrix([[1,0,-1],[1,0,-1],[1,0,-1]])
    hor_filt = np.matrix([[1,1,1],[0,0,0],[-1,-1,-1]])
    diag_tl_br = np.matrix([[0,2,1],[-2,0,2],[1,-2,1]])
    diag_bl_tr = np.matrix([[1,2,0],[2,0,-2],[0,-2,1]])
    
    
    flattened_training_data = []
    training_label_arrays = []
    n_mat = 100
    for i in range(n_mat):  # len(raw_training_data)
        conv_data = []
        conv_data.append(convoluteInputData(raw_training_data[i], n_conv_laps, vert_filt).flatten())
        conv_data.append(convoluteInputData(raw_training_data[i], n_conv_laps, hor_filt).flatten())
        conv_data.append(convoluteInputData(raw_training_data[i], n_conv_laps, diag_tl_br).flatten())
        conv_data.append(convoluteInputData(raw_training_data[i], n_conv_laps, diag_bl_tr).flatten())
        
        flattened_training_data.append(conv_data)
        
        label_as_array = [0.0] * 10
        label_as_array[raw_training_labels[i]] = 1.0
        training_label_arrays.append(label_as_array)
        if  i%(n_mat/10) == 0:
            print(str(int(100*i/n_mat)) + '% of the training data convoluted!')
        
    print('100% of the training data convoluted!\n')
    flattened_validating_data = []
    for i in range(n_mat):  # len(raw_validating_data)
        conv_data = []
        conv_data.append(convoluteInputData(raw_validating_data[i], n_conv_laps, vert_filt).flatten())
        conv_data.append(convoluteInputData(raw_validating_data[i], n_conv_laps, hor_filt).flatten())
        conv_data.append(convoluteInputData(raw_validating_data[i], n_conv_laps, diag_tl_br).flatten())
        conv_data.append(convoluteInputData(raw_validating_data[i], n_conv_laps, diag_bl_tr).flatten())
        
        flattened_validating_data.append(conv_data)
        
        if  i%(n_mat/10) == 0:
            print(str(int(100*i/n_mat)) + '% of the validation data convoluted!')
            
    print('100% of the validation data convoluted!\n')
    print('Data sucessfully convoluted!')
    return flattened_training_data, training_label_arrays, flattened_validating_data, raw_validating_labels#validating_label_arrays
    

def convoluteInputData(data, rem_laps, conv_filter):
    if len(data) < 5 or rem_laps == 0:   # Smallest allowed matrix size for one Conv revolution is 5x5.
        return data

    # Filtrera
    #timer_conv_filt = timeit.timeit(stmt = lambda: convFilter(data, conv_filter), number = 1)
    result_mat = convFilter(data, conv_filter)
    # Normalisera
    #timer_normalize = timeit.timeit(stmt = lambda: normalize(result_mat), number = 1)
    result_mat = normalize(result_mat)
    # Poola
    #timer_pool = timeit.timeit(stmt = lambda: pool(result_mat), number = 1)
    result_mat = pool(result_mat)
    
    return convoluteInputData(result_mat, rem_laps - 1, conv_filter)
   
def convFilter(data, conv_filter):
    result_mat = np.zeros((len(data) - len(conv_filter) + 1, len(data) - len(conv_filter) + 1))
    for i in range(len(data) - len(conv_filter) + 1):
        for j in range(len(data) - len(conv_filter) + 1):
            temp = 0
            for k in range(len(conv_filter)):
                for l in range(len(conv_filter)):
                    temp += data[i+k,j+l] * conv_filter[k,l]
            result_mat[i,j] = temp
    
    return result_mat

def normalize(result_mat):
    map_sig_norm = np.vectorize(sig_norm)
    return map_sig_norm(result_mat)


def pool(result_mat):
    s = 2   # Stride length
    if len(result_mat)%2 == 0:
        result_mat_2 = np.zeros((int(len(result_mat)/2), int(len(result_mat)/2)))
        odd_size = False
    elif len(result_mat)%2 == 1:
        result_mat_2 = np.zeros((int(len(result_mat)/2 + 1), int(len(result_mat)/2 + 1)))
        odd_size = True
    else:
        print('Something went wrong! Check method applyFilter!')
        return None
    
    for i in range(len(result_mat_2)):
        for j in range(len(result_mat_2)):
            temp = []
            k_lim = 2
            l_lim = 2
            if odd_size and i == len(result_mat_2) - 1:
                k_lim = 1
            if odd_size and j == len(result_mat_2) - 1:
                l_lim = 1
            for k in range(k_lim):
                for l in range(l_lim):
                    temp.append(result_mat[s*i+k,s*j+l])
            result_mat_2[i,j] = max(temp)
    return result_mat_2


def sig_norm(x):
    return 1 / (1 + math.exp(-x))  


main()
print('Execution completed!')
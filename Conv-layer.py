# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:22:24 2020

@author: filip
"""
import math
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def getConvolutedData(n_conv_laps):
    # move to separate method
    print('Opening MNIST data base...')
    map_divide_by_255 = np.vectorize(divide_by_255)
    
    train_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_training_data = idx2numpy.convert_from_file(train_data_file)
    raw_training_data = map_divide_by_255(raw_training_data)
    
    train_label_file = open('train-labels.idx1-ubyte', 'rb')
    raw_training_labels = idx2numpy.convert_from_file(train_label_file)
    
    #validating_data_file = open('train-images.idx3-ubyte', 'rb')
    #raw_validating_data = idx2numpy.convert_from_file(validating_data_file)
    #raw_validating_data = map_divide_by_255(raw_validating_data)
    
    #validating_label_file = open('train-labels.idx1-ubyte', 'rb')
    #raw_validating_labels = idx2numpy.convert_from_file(validating_label_file)  
    
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
    
    """
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
    """
    
    print('100% of the validation data convoluted!\n')
    print('Data sucessfully convoluted!\n')
    return flattened_training_data, training_label_arrays, raw_training_data[0] #, flattened_validating_data, raw_validating_labels#validating_label_arrays
    

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

def divide_by_255(x):
    return x/255


n_conv_laps = 2;
conv_data, labels, raw_data = getConvolutedData(n_conv_laps)


plt.matshow(raw_data)
plt.title('Original picture:')

plt.matshow(conv_data[0])
plt.title('Convoluted data:')

print('The convoluted data is flattened to fit the input layer of ANN')
print('Each rown in the convoluted picture correspond to one filter')
print('0th row correspond to VERTICAL edge detection filter')
print('1st row correspond to HORIZONTAL edge detection filter')
print('2nd row correspond to DIAGONAL (top left to bottom right) edge detection filter')
print('3nd row correspond to DIAGONAL (top right to bottom left) edge detection filter')

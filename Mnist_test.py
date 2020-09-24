# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:25:19 2020

@author: filip
"""

import idx2numpy
import numpy as np

def main():
    train_data_file = open('train-images.idx3-ubyte', 'rb')
    raw_training_data = idx2numpy.convert_from_file(train_data_file)
    
    label_data_file = open('train-labels.idx1-ubyte', 'rb')
    raw_label_data = idx2numpy.convert_from_file(label_data_file)
    

    flattened_data = []
    label_arrays = []

    for i in range(len(raw_training_data)):
        flattened_data.append(raw_training_data[i].flatten())
        target_array = [0.0] * 10
        target_array[raw_label_data[i] - 1] = 1.0
        label_arrays.append(target_array)
    print(len(label_arrays))

main()
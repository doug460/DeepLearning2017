'''
Created on Sep 11, 2017

@author: DougBrownWin

variabls for back prop
'''

import numpy as np

# create y matrix output
y = np.matrix('1 1 0 1 0 1 1 0 1')

# input matrix with two atributes
x = np.matrix('5 6 1 7 3 5 6 1 3; 1 2 5 3 7 2 1 9 2')

# hidden layers
hidden_layers = 3

# neurons in hidden layer
hidden_neurons = 5

# step size for error adjustment
epsilon = 1



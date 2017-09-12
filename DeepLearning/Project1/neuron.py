'''
Created on Sep 11, 2017

@author: DougBrownWin
'''

import variables as varis
import numpy as np
from numba.types import none

class Neuron(object):
    '''
    classdocs
    '''


    def __init__(self, numOf_weights):
        '''
        Constructor
        '''
        # define output of neuron
        self.out = 0
        
        # store info before going through activation function
        self.sum = 0
        
        # get weights going into neuron!!
        self.weights = []
        
        # delta for this neuron
        self.delta = 0
        
        # if last layer, weights are to output layer
        # initialize weights to 1
        for indx in range(0, numOf_weights):
            self.weights.append(1)
            
        # for updating weights
        self.weights_updated = self.weights
        
    # set output of neuron
    def setOut(self, value):
        self.out = value
        
    # calculate new output based on previous layer
    def calcOut(self, previous_layer):
        # last weight is a bias
        # add bias
        sum = self.weights[len(self.weights) - 1]
        
        # sum up weights multiplied by output of neurons
        for indx in range(0,len(previous_layer.neurons)):
            sum += previous_layer.neurons[indx].out * self.weights[indx] 
            
        # store summ
        self.sum = sum
        
        # calculate sigmoid
        self.out = 1 / (1 + np.exp(-sum))


    # def calcDelta for last layer
    def calcDelta_lastLayer(self, y, y_out):
        self.delta = (np.subtract(y, y_out) * (np.exp(-self.sum) / (1+np.exp(-self.sum))**2))

    # calc chagne in weights for last layer
    def calcWeights(self, previous_layer):
        
        # calcualte updated weights
        weights = []
        neurons_prev = previous_layer.neurons
        for indx in range(0, len(previous_layer.neurons)):
            weights.insert(indx, self.delta * neurons_prev[indx].out)
    
        # if this is running on first data set
        if(len(self.weights_updated) == 0):
            self.weights_updated =  np.multiply(weights, (-1 * varis.epsilon))
        else:
            self.weights_updated +=  np.multiply(weights, (-1 * varis.epsilon))
        
    
    # calc delta for other hidden layers
    # delta for each neuron in next layer
    def calcDelta(self, layer_next, current_neuron_indx):
        # g prime value
        g_prime = (np.exp(-self.sum) / (1+np.exp(-self.sum))**2)
         
        # step through each neuron
        summation = 0
        for indx in range(0, len(layer_next.neurons)):
            neuron_next = layer_next.neurons[indx]
            summation  += neuron_next.weights[current_neuron_indx] * neuron_next.delta
        
        self.delta = g_prime * summation
        
    
    # update weights of system
    def updateWeights(self):
        self.weights = self.weights_updated
                








'''
Created on Sep 11, 2017

@author: DougBrownWin
'''

import variables as varis
import numpy as np

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
        self.delta = []
        
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
    def calcDelta_lastLayer(self, y_out):
        self.delta = (np.subtract(varis.y, y_out) * (np.exp(-self.sum) / (1+np.exp(-self.sum))**2))

    # calc chagne in weights for last layer
    def calcWeights_lastLayer(self, previous_layer):
        
        # calcualte updated weights
        weights = []
        neurons = previous_layer.neurons
        for indx in range(0, len(previous_layer.neurons)):
            weights.insert(indx, self.delta[0, indx] * neurons[indx].out)
    
        self.weights_updated =  np.multiply(weights, (-1 * varis.epsilon))








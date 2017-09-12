'''
Created on Sep 11, 2017

@author: DougBrownWin

Program to work on back propogation


'''

import numpy as np
import variables as varis
from layer import Layer

def calcOut(layers):
    # calculate output
    y_out = []
    for column in range(0,varis.x.shape[1]):
        # select all of column 1
        # set input layer
        layers[0].setOut(varis.x[:,column])
         
         
        # calculate layers skipping input layer
        for layer in range(1, len(layers)):
            layers[layer].calcOut(layers[layer - 1])         
 
        # get output
        last_neurons = layers[len(layers) - 1].neurons
        for neuron in last_neurons:
            y_out.append(neuron.out)
        
    # return schtuff
    return(y_out)

# calcualte weights for function
def updateWeights(layers, y_out):
    # start with last layer
    for neuron in layers[len(layers) - 1].neurons:
        neuron.calcDelta_lastLayer(y_out)
        neuron.calcWeights_lastLayer(layers[len(layers)-2])
    
    

def calcError(y_out):
    y_size = len(y_out)
    error = 1/(2*y_size) * np.sum(np.square(np.subtract(varis.y, y_out)))
    return(error)
    

if __name__ == '__main__':
    pass

    # layers
    layers = []

    # +2 for input/output layers
    for indx in range(0, varis.hidden_layers + 2):
        layer = Layer(indx)
        layers.append(layer)
    
    # calculate output
    y_out = calcOut(layers)
    print(y_out)
    
    # calculate error
    error = calcError(y_out)
    print('error is %f ' % (error))
    
    updateWeights(layers, y_out)
    
    
    
    
     

    
    
    
    
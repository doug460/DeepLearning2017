'''
Created on Sep 11, 2017

@author: DougBrownWin

Program to work on back propogation


'''

import numpy as np
import variables as varis
from layer import Layer

# calcualte weights for function
def backPropogation(layers, y, y_out):
    # start with last layer
    for indx in range(0,len(layers[len(layers) - 1].neurons)):
        neuron = layers[len(layers)-1].neurons[indx]
        
        neuron.calcDelta_lastLayer(y[indx], y_out[indx])
        previous_layer = layers[len(layers) - 2]
        neuron.calcWeights(previous_layer)
    
    # go backwards through rest of layers updating neurons
    # start at second to last layer
    # stop at second layer
    for indx in range(1, len(layers) - 1):
        # indx for counting backwards
        layer_indx = len(layers) - 1 - indx
         
        # current layer
        layer = layers[layer_indx]
        
        # previous layer
        previous_layer = layers[layer_indx - 1]
         
        # step through each neuron and update weights
        for neuron_indx in range(0,len(layer.neurons)):
            neuron = layer.neurons[neuron_indx]
            
            # pass next layer 
            # pass currnet neuron indx
            neuron.calcDelta(layers[layer_indx + 1], neuron_indx)
            
            # calculate weight changes
            neuron.calcWeights(previous_layer)
            
            

def calcOut(layers):
    # initialize change wegiths to zero in neurons
    for layer in layers:
        for neuron in layer.neurons:
            neuron.weights_updated = []
    
    # calculate output
    y_out = np.zeros(varis.y.shape)
    for column in range(0,varis.x.shape[1]):
        # select all of column 1
        # set input layer
        layers[0].setOut(varis.x[:,column])
         
         
        # calculate layers skipping input layer
        for layer in range(1, len(layers)):
            layers[layer].calcOut(layers[layer - 1])         
 
        # get output
        last_neurons = layers[len(layers) - 1].neurons
        for indx in range(0,len(last_neurons)):
            neuron = last_neurons[indx]
            out = neuron.out
        
            y_out[indx, column] = out
            
        # need to track the weight changes
        backPropogation(layers, varis.y[:,column], y_out[:, column])
        
    # return schtuff
    return(y_out)

# push updated weights into system
def updateWeights(layers):
    for layer in layers:
        for neuron in layer.neurons:
            neuron.updateWeights()  
        
    

def calcError(y_out):
    y_size = y_out.shape[1]    
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
    
    # update weights
    updateWeights(layers)
    
    # calculate error
    error = calcError(y_out)
    print('error is %f ' % (error))
    
    for indx in range(0, 20):
        # rerun schtuff
        y_out = calcOut(layers)
        error = calcError(y_out)
        updateWeights(layers)
        print('error is %f ' % (error))
    
    
    
    
     

    
    
    
    
'''
Created on Sep 11, 2017

@author: DougBrownWin
'''

from neuron import Neuron
import variables as varis


class Layer(object):
    '''
    classdocs
    '''


    def __init__(self, layer_indx):
        '''
        Constructor
        '''
        self.layer_indx = layer_indx
        
        # create neurons
        self.neurons = []
        
        
        # if input layer
        if(layer_indx == 0):
            for k in range(0, varis.x.shape[0]):
                numOf_weights = 0
                neuron = Neuron(numOf_weights)
                self.neurons.append(neuron)
            
        # if middle layers
        elif(layer_indx <= varis.hidden_layers):
            
            if(layer_indx == 1):
                # first hidden layer
                
                # +1 for bias weight
                numOf_weights = varis.x.shape[0] + 1
                neuron = Neuron(numOf_weights)
                self.neurons.append(neuron)
                
            else:  
                # rest of the layers
                for k in range(0,varis.hidden_neurons):
                    
                    # +1 for bias weight
                    numOf_weights = varis.hidden_neurons + 1
                    neuron = Neuron(numOf_weights)
                    self.neurons.append(neuron)
                    
            
        # if last layer
        else:
            for k in range(0,varis.y.shape[0]):
                numOf_weights = varis.hidden_neurons
                neuron = Neuron(numOf_weights)
                self.neurons.append(neuron)


    # define output for each neuron
    def setOut(self, array):
        for indx in range(0,array.shape[0]):
            self.neurons[indx].setOut(array[indx,0])
            
        
    # calculate output
    def calcOut(self, previous_layer):
        # send layer to each neuron to calculate output
        for indx in range(0, len(self.neurons)):
            self.neurons[indx].calcOut(previous_layer)
        






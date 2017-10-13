'''
Created on Oct 10, 2017

@author: dabrown
'''

import sys
from random import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def square(array):
    return tf.square(array)

def loss(array):
    term1 = tf.square(tf.reduce_sum(array) - 10)
    term2 = tf.square(tf.reduce_prod(array) - 24)
    
    return term1 + term2
    

if __name__ == '__main__':
    pass

    array = tf.Variable([[2.0,5],[1,4]])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    
    
    loss_op = loss(array)
    train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_op) 

    sess.run(init)
    
    for n in range(2000):
        sess.run(train)
     
    print(sess.run(array))
    print(sess.run(loss_op))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
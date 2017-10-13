'''
Created on Oct 13, 2017

@author: dabrown
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

from tensorflow.examples.tutorials.mnist import input_data
from astropy.convolution.convolve import convolve
data = input_data.read_data_sets('data/MNIST', one_hot=True)

# batch size
train_batch_size = 64


# filters and fully connected stuff
filter_size1 = 5
num_filter1 = 16

filter_size2 = 5
num_filter2 = 36

fc_size = 128

# basically just compressing data to be index values instead of one hot encoding
data.test.cls = np.argmax(data.test.labels, axis = 1)

# image stuff
image_size = 28
image_flat = image_size * image_size

image_shape = (image_size, image_size)

# number of color channels
num_channels = 1

num_classes = 10

sess = tf.Session()

# functions to get weights and biases
def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev = 0.05))

def get_biasses(length):
    return tf.Variable(tf.constant(0.05, shape = [length]))

# create a new convolutional layer
def get_convolveLayer(inLayer, num_input_channels, filter_size, filter_num, use_pooling = True):
    shape = [filter_size, filter_size, num_input_channels, filter_num]
    
    # get weights and biases
    weights = get_weights(shape=shape)
    
    biases = get_biasses(filter_num)
    
    layer = tf.nn.conv2d(inLayer, filter = weights, strides = [1,1,1,1], padding='SAME')
    layer += biases
    
    if use_pooling:
        layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
    layer = tf.nn.relu(layer)
    
    return(layer, weights)

# need to flatten layer for fc layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    
    num_features = layer_shape[1:4].num_elements()
    
    layer_flat = tf.reshape(layer, shape = [-1, num_features])
    
    return layer_flat, num_features

# also need fully connected layer
def get_fc_layer(inLayer, num_inputs, num_outputs, use_relu = True):
    weights = get_weights(shape=[num_inputs, num_outputs])
    biases = get_biasses(length = num_outputs)
    
    layer = tf.matmul(inLayer,weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

# optimize stuff
# count iterations
total_iterations = 0
def optimize(num_iterations):
    start_time = time.time()
    
    global total_iterations
    
    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        sess.run(optimizer, feed_dict=feed_dict_train)
        print('accuracy is: ',sess.run(accuracy, feed_dict=feed_dict_train),'\n')
    
    total_iterations += num_iterations
    
    end_time = time.time()
    
    time_dif = end_time - start_time
    print("time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    

if __name__ == '__main__':
    pass

    # get placeholders for data
    x = tf.placeholder(dtype=tf.float32, shape = [None, image_flat], name = 'x')
    
    # for going into convolutional layer
    x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])
    
    # get y true values and compressed values
    y_true = tf.placeholder(dtype = tf.float32, shape=[None,num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    
    # create first convolutional layer
    layer_conv1, weights1 = get_convolveLayer(inLayer=x_image, num_input_channels=num_channels, filter_size=filter_size1, filter_num=num_filter1)
    
    # get layer 2
    layer_conv2, weights2 = get_convolveLayer(inLayer = layer_conv1, num_input_channels=num_filter1, filter_size=filter_size2, filter_num=num_filter2) 
    
    # flatten output of convlution
    layer_flat, num_features = flatten_layer(layer_conv2)
    
    # get fully connected layer
    layer_fc1 = get_fc_layer(layer_flat, num_inputs=num_features, num_outputs=fc_size)
    
    # want output of 10 for one hot encoded
    layer_fc2 = get_fc_layer(inLayer=layer_fc1, num_inputs=fc_size, num_outputs=num_classes)
    
    # want what is predicted
    y_pred = tf.nn.softmax(layer_fc2)
    
    # get compressed
    y_pred_cls = tf.argmax(y_pred, axis=1)
    
    # need cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=layer_fc2)
    cost = tf.reduce_mean(cross_entropy)
    
    # define an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # initialize stuff
    sess.run(tf.global_variables_initializer())
    
    optimize(num_iterations=500)
    
    
    
    
    
    
    
    

































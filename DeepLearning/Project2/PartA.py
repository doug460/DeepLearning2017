'''
Created on Sep 27, 2017

@author: dabrown

bassically using network A and display one of each image
'''

import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import tensorflow as tf
import time, shutil, os
from fdl_examples.datatools import input_data
import matplotlib.pyplot as plt

# read in MNIST data --------------------------------------------------
mnist = input_data.read_data_sets("../../data/", one_hot=True)


# run network ----------------------------------------------------------

# Parameters
learning_rate = 0.01
training_epochs = 5 # NOTE: you'll want to eventually change this 
batch_size = 100
display_step = 1


def inference(x,W,b):
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("output", output)
    
    return output

def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row 
    # into a single value. In general, reduction along axis i 
    # collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, axis=1)
     
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):

    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy

def saveImage(array, number):
    matrix = np.reshape(array, (28,28))
    plt.imshow(matrix, cmap = 'gray_r')
    plt.title(number)
    buf = 'PartA Images/origonals/data_%d.png' % (number)
    plt.savefig(buf)
    
def saveWeights(W):
    w_out = sess.run(W)
    w_out_images = np.reshape(w_out, (784,10))
    for indx in range(0,10):
        image = w_out[:,indx]
        matrix = np.reshape(image, (28,28))
        plt.imshow(matrix)
        
        buf = 'PartA Images/weights_a/data_%d.png' % (indx)
        title = 'Number %d' % (indx)
        plt.title(title)
        plt.savefig(buf)
    

if __name__ == '__main__':
    if os.path.exists("logistic_logs/"):
        shutil.rmtree("logistic_logs/")

    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

        init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", [784, 10],
                             initializer=init)    
        b = tf.get_variable("b", [10],
                             initializer=init)

        output = inference(x,W,b)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        train_op = training(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("logistic_logs/",
                                            graph_def=sess.graph_def)

        
        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        
        minix, miniy = mnist.train.next_batch(30)        
        # used this to see what index was what for the images
        # print(miniy)
        
        # save data
        saveImage(minix[7],  0)
        saveImage(minix[4],  1)
        saveImage(minix[13], 2)
        saveImage(minix[1],  3)
        saveImage(minix[2],  4)
        saveImage(minix[27], 5)
        saveImage(minix[3],  6)
        saveImage(minix[14], 7)
        saveImage(minix[5],  8)
        saveImage(minix[8],  9)

        # Training cycle
        for epoch in range(training_epochs):
 
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
 
                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
 
                print("Validation Error:", (1 - accuracy))
 
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                #summary_writer.add_summary(summary_str, sess.run(global_step))
 
                #saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
            
 
        saveWeights(W)
        
        
 
        print("Optimization Finished!")
         
        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
 
        print("Test Accuracy:", accuracy)
#         

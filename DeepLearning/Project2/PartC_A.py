'''
Created on Sep 27, 2017

@author: dabrown

second part of project two. for part A
basically no hidden layers
'''

import sys
from scipy.ndimage.interpolation import zoom
from random import random
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
training_epochs = 50 # NOTE: you'll want to eventually change this 
batch_size = 100
display_step = 1

saveDir = 'PartC Stuff/'


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

# array is image length (784) * batch length (100)
# so 784 * 100
def applyScaling(array):
    
    for indx in range(0, batch_size):
        # seperate an single image array and put it in matrix form
        single_array = array[indx]
        single_image = np.reshape(single_array, (28,28))
        
        # scale image
        scale_amount = random() * 0.5 + 0.5
        single_image_scale = zoom(single_image, zoom = scale_amount)
        
        matrix_padded = np.zeros((28,28))
        dim = single_image_scale.shape[0]
        start = int(28/2 - dim/2)
        matrix_padded[start:start + dim, start:start + dim] = single_image_scale
        
        # return image back to array and replace in images_array
        single_array_scale = np.reshape(matrix_padded, (784))
        array[indx] = single_array_scale
    
    return(array)

# function to save the weights
def saveWeights(W):
    w_out = sess.run(W)
    w_out_images = np.reshape(w_out, (784,10))
    for indx in range(0,10):
        image = w_out[:,indx]
        matrix = np.reshape(image, (28,28))
        plt.imshow(matrix)
        
        buf = saveDir + 'weights_a/data_%d.png' % (indx)
        title = 'Number %d' % (indx)
        plt.title(title, fontsize=25)
        plt.savefig(buf)

def getY(y):
    return(tf.argmax(y,1))

def getOut(output):
    return(tf.argmax(output,1))

# create the confusion matrix
# acutal on rows
# predicted on columns
def createConfusion(outputs, y):
    matrix = np.zeros((10,10))
    for indx in range(0,len(outputs)):
        row = y[indx]
        column = outputs[indx]
        matrix[row,column] += 1
        
    plt.close()
    plt.figure(figsize = (15,15))
    labels = ('0','1','2','3','4','5','6','7','8','9')
    tb = plt.table(cellText=matrix, loc=(0,0), cellLoc='center', rowLabels=labels,colLabels=labels)
    
    tc = tb.properties()['child_artists']
    for cell in tc: 
        cell.set_height(1.0/11)
        cell.set_width(1.0/11)
    
    plt.title('Confusion Matrix', fontsize=25)
    plt.xlabel('Predicted', fontsize=25)
    plt.ylabel('Actual', fontsize=25)
    
    tb.set_fontsize(14)
    
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(saveDir + 'confusionMatrixC_A.png')
    
    
    

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
        
        gety = getY(y)
        getout = getOut(output)

        
        minix, miniy = mnist.train.next_batch(30)        
        # used this to see what index was what for the images
        # print(miniy)
         
        # save data
        # basically go through and save one of each image
        test = minix[1] # this is for number 3
        matrix = np.reshape(test, (28,28))
        plt.imshow(matrix)
        plt.title('Origonal 3', fontsize=25)
        plt.savefig(saveDir + 'origonal_3.png')
         
        matrix_scale = zoom(matrix, 0.6)
        matrix_padded = np.zeros((28,28))
        dim = matrix_scale.shape[0]
        start = int(28/2 - dim/2)
        matrix_padded[start:start + dim, start:start + dim] = matrix_scale
        plt.imshow(matrix_padded)
        plt.title('3 scaled to 60%', fontsize=25)
        plt.savefig(saveDir + 'scaled_3.png')
        
        
        # want to graph accuracies
        accuracies = []
   
        # Training cycle
        for epoch in range(training_epochs):
   
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                 
                minibatch_x = applyScaling(minibatch_x)
                 
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
   
                accuracy = sess.run(eval_op, feed_dict={x: applyScaling(mnist.validation.images), y: mnist.validation.labels})
   
                print("Validation Error:", (1 - accuracy))
   
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                #summary_writer.add_summary(summary_str, sess.run(global_step))
   
                #saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
             
            # save accuries over time
            accuracies.append(sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
          
   
        print("Optimization Finished!")
           
        accuracy = sess.run(eval_op, feed_dict={x: applyScaling(mnist.test.images), y: mnist.test.labels})
         
        # get outputs for comparison
        # shape is ?,10
        y_temp = sess.run(gety, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        out_temp = sess.run(getout, feed_dict={x: mnist.test.images, y: mnist.test.labels})
         
        # create and save confusion matrix
        createConfusion(out_temp, y_temp)
         
        # save the weights
        saveWeights(W)
         
   
        print("Test Accuracy:", accuracy)   
         
        # plot accuracies vs iterations
        plt.clf()
        x = range(0, len(accuracies))
        plt.plot(x, accuracies)
        plt.title('Accuracy vs. Iteration', fontsize=25)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.savefig(saveDir + 'partC_A_acc.png')
         
         
        file  = open(saveDir + "partC_A_info.txt",'w')
        buf = "Accuracy was %f\n" % (accuracy)
        file.write(buf)
        file.close()
        
        
        

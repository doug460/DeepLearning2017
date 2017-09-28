import sys
sys.path.append('../../')
sys.path.append('../')
from fdl_examples.datatools import input_data
mnist = input_data.read_data_sets("../../data/", one_hot=True)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import time, shutil, os

saveDir = 'PartA Stuff/'

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameters
learning_rate = 0.01
training_epochs = 2 # NOTE: you'll want to eventually change this 
batch_size = 100
display_step = 1

#def layer(input, weight_shape, bias_shape):
def layer(input, W, b):
    return tf.nn.relu(tf.matmul(input, W) + b)

#def inference(x,W1,b1,W2,b2):
def inference(x):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, W1,b1)
     
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1,W2,b2)
     
    with tf.variable_scope("output"):
        #output = layer(hidden_2, [n_hidden_2, 10], [10])
        output = layer(hidden_2, W3,b3)

    return output

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)    
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
    tf.summary.scalar("validation", accuracy)
    return accuracy


# save the weights of the first images
def saveWeights1(W):
    w_out = sess.run(W)
    w_out_less = w_out[:,0:10]
    print("w_less", w_out_less.shape)
    w_out_images = np.reshape(w_out_less, (784,10))
    for indx in range(0,10):
        image = w_out[:,indx]
        matrix = np.reshape(image, (28,28))
        plt.imshow(matrix)
        
        buf = saveDir + 'weights_b/w1_data_%d.png' % (indx)
        title = 'Node %d' % (indx)
        plt.title(title)
        plt.savefig(buf)

# def saveWeights2(W):
#     w_out = sess.run(W)
#     w_out_images = np.reshape(w_out, (256,10))
#     for indx in range(0,10):
#         image = w_out[:,indx]
#         matrix = np.reshape(image, (16,16))
#         plt.imshow(matrix)
#         
#         buf = 'PartA Images/weights_b/w2_data_%d.png' % (indx)
#         plt.savefig(buf)
#         
# def saveWeight3(W):
#     w_out = sess.run(W)
#     w_out_images = np.reshape(w_out, (784,10))
#     for indx in range(0,10):
#         image = w_out[:,indx]
#         matrix = np.reshape(image, (28,28))
#         plt.imshow(matrix)
#         
#         buf = 'PartA Images/weights_b/w3_data_%d.png' % (indx)
#         plt.savefig(buf)

if __name__ == '__main__':
    
    if os.path.exists("mlp_logs/"):
        shutil.rmtree("mlp_logs/")

    with tf.Graph().as_default():

        with tf.variable_scope("mlp_model"):

            x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
            y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

            weight_init1 = tf.random_normal_initializer(stddev=(2.0/784)**0.5)
            weight_init2 = tf.random_normal_initializer(stddev=(2.0/n_hidden_1)**0.5)
            weight_init3 = tf.random_normal_initializer(stddev=(2.0/n_hidden_2)**0.5)
            bias_init = tf.constant_initializer(value=0.)
            
            W1 = tf.get_variable("W1", [784, n_hidden_1],
                                initializer=weight_init1)
            b1 = tf.get_variable("b1", [n_hidden_1],
                                initializer=bias_init)
            W2 = tf.get_variable("W2", [n_hidden_1, n_hidden_2],
                                initializer=weight_init2)
            b2 = tf.get_variable("b2", [n_hidden_2],
                                initializer=bias_init)
            W3 = tf.get_variable("W3", [n_hidden_2, 10],
                                initializer=weight_init3)
            b3 = tf.get_variable("b3", [10],
                                initializer=bias_init)

            #output = inference(x,W1,b1,W2,b2)
            output = inference(x)

            cost = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)

            eval_op = evaluate(output, y)

            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            sess = tf.Session()

            summary_writer = tf.summary.FileWriter("mlp_logs/",
                                                graph_def=sess.graph_def)

            
            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            # saver.restore(sess, "mlp_logs/model-checkpoint-66000")

            # save accuracies 
            accuracies = []

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
                    summary_writer.add_summary(summary_str, sess.run(global_step))

                    saver.save(sess, "mlp_logs/model-checkpoint", global_step=global_step)
                    
                
                
                accuracies.append(sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

            saveWeights1(W1)

            print("Optimization Finished!")


            accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})

            print("Test Accuracy:", accuracy)
            
            # plot accuracies vs iterations
            plt.clf()
            x = range(0, len(accuracies))
            plt.plot(x, accuracies)
            plt.title('Accuracy vs. Iteration')
            plt.ylabel('Accuracy')
            plt.xlabel('Iteration')
            buf = saveDir + 'partA_B_acc.png'
            plt.savefig(buf)
            
            buf = saveDir + 'partA_B_info.txt'
            file  = open(buf,'w')
            buf = "Accuracy was %f\n" % (accuracy)
            file.write(buf)
            file.close()


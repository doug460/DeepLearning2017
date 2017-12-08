'''
Created on Nov 8, 2017

@author: dabrown

This is my attempt at a RNN 
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from datetime import timedelta
import MainDir
import pickle
import cmath

with open(MainDir.dirData + 'data.pickle', 'rb') as f:
    train_x, train_y, train_ref, test_x, test_y, test_ref, dsp_x, dsp_y, dsp_ref = pickle.load(f)

num_iterations = 5000

print_every = 10

# Convolutional Layer 1.
filter_size_raw = 10
num_filters_raw = 16         # There are 16 of these filters.

filter_size_ref = 10
num_filters_ref = 16

# size of input raw layer
fc_size_raw = 500
fc_size_ref = 500

# # Convolutional Layer 2.
# filter_size2 = 20         # Convolution filters are 5 x 5 pixels.
# num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc1_size = 500             # Number of neurons in fully-connected layer.
fc2_size = 500

# We know that MNIST images are 28 pixels in each dimension.
img_size = 500


train_batch_size = 100  

test_batch_size = 256



# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (1, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# define epsilon!
epsilon = 0.001

# track iterations throughout run
total_iterations = 1

error_track = []

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=False):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [1, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
#     layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

    
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=False): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def getMini_batch(x,y, ref, batch_size):
    length = x.shape[0]
    
    
    
    for i in range(1,batch_size):
        indx = random.randint(0,length-1)
        xsub = np.expand_dims(x[indx], axis = 0)
        refsub = np.expand_dims(ref[indx], axis = 0)
        ysub = np.expand_dims(y[indx], axis = 0)
        
        if i == 1:
            xMini = xsub
            refMini = refsub
            yMini = ysub
        else:
            xMini = np.vstack((xMini,xsub))
            refMini = np.vstack((refMini,refsub))
            yMini = np.vstack((yMini,ysub))
    
    return xMini, yMini, refMini


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations, error_track

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations + 1):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_batch, ref_batch = getMini_batch(train_x, train_y, train_ref, train_batch_size)
        
        

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_batch,
                           ref: ref_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        sess.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % print_every == 0:
            # Calculate the accuracy on the training-set.
            err = sess.run(error, feed_dict=feed_dict_train)
            error_track.append(err)

            # Print it.
            print("Optimization Iteration: %d, Training Error is: %.2f" % (i, err))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Split the test-set into smaller batches of this size.


if __name__ == '__main__':
    pass

    x = tf.placeholder(tf.float32, shape=[None, 1, img_size_flat, 1], name='x')
    ref = tf.placeholder(tf.float32, shape = [None, 1, img_size_flat, 1], name='ref')
    y_true = tf.placeholder(tf.float32, shape=[None, img_size], name='y_true')
    
    
    layer_conv_raw, weights_conv1 = new_conv_layer(input=x,
                   num_input_channels=num_channels,
                   filter_size=filter_size_raw,
                   num_filters=num_filters_raw,
                   use_pooling=False)
    
    layer_conv_ref, weights_conv1 = new_conv_layer(input=ref,
                   num_input_channels=num_channels,
                   filter_size=filter_size_ref,
                   num_filters=num_filters_ref,
                   use_pooling=False)
    
#     layer_conv1 = tf.nn.dropout(layer_conv1, keep_prob= 0.75)
    
#     layer_conv2, weights_conv2 = \
#     new_conv_layer(input=layer_conv1,
#                    num_input_channels=num_filters1,
#                    filter_size=filter_size2,
#                    num_filters=num_filters2,
#                    use_pooling=True)
    
    layer_flat_raw, num_features_raw = flatten_layer(layer_conv_raw)
    
    layer_raw = new_fc_layer(input = tf.squeeze(x), 
                             num_inputs = img_size,
                             num_outputs = fc_size_raw,
                             use_relu = False)
    
    layer_flat_ref, num_features_ref = flatten_layer(layer_conv_ref)
    
    layer_ref = new_fc_layer(input = tf.squeeze(ref), 
                             num_inputs = img_size,
                             num_outputs = fc_size_ref,
                             use_relu = False)
    
    # stack the layers
    layer_stacked = tf.concat([layer_flat_raw, layer_raw, layer_flat_ref, layer_ref], axis = 1)
    
    layer_fc1 = new_fc_layer(input=layer_stacked,
                         num_inputs=num_features_raw + fc_size_raw + num_features_ref + fc_size_ref,
                         num_outputs=fc1_size,
                         use_relu=True)
    
#     layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob=0.75)
    
    layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc1_size,
                         num_outputs=fc2_size,
                         use_relu=True)
    
    layer_out = new_fc_layer(input=layer_fc2,
                         num_inputs=fc2_size,
                         num_outputs=img_size,
                         use_relu=False)
    
    error = tf.sqrt(tf.reduce_mean(tf.square(layer_out - y_true)))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=epsilon).minimize(error)     
    
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
    
        optimize(num_iterations) # We already performed 1 iteration above.
        
        feed_dict = {x: dsp_x, ref: dsp_ref, y_true: dsp_y}
        output = sess.run(layer_out, feed_dict = feed_dict)
        error_num = sess.run(error, feed_dict = feed_dict)
         
        # flatten output
        output = np.reshape(output, (len(output)*img_size))
 
        # get data flat
        dsp_xflat = np.reshape(dsp_x, (len(dsp_x)*img_size))
        dsp_yflat = np.reshape(dsp_y, (len(dsp_y)*img_size))
          
        # plot x
        fig = plt.figure()
        plt.plot(dsp_xflat)
        plt.title('Noisy Data')
        plt.savefig(MainDir.dirData + 'noisyData.png')
          
        # plot y
        fig = plt.figure()
        plt.plot(dsp_yflat)
        plt.title('Original Signal')
        plt.savefig(MainDir.dirData + 'origonal.png')
          
        # plot output
        fig = plt.figure()
        plt.plot(output)
        plt.title("Output Data")
        plt.savefig(MainDir.dirData + 'output.png')
        
        # get plot of error vs time
        fig = plt.figure()
        plt.plot(error_track)
        plt.title('Error vs Iterations')
        plt.xlabel('n/%d' % (print_every))
        plt.ylabel('RMS error')
        plt.savefig(MainDir.dirData + 'error.png')
        
        buf = 'ending error is %f%%\n' % (error_num*100)
        file = MainDir.dirData + 'info.txt'
        file  = open(file,'w')
        file.write(buf)
          
        plt.show()
        
    
    
        # save output to commputer
#         saver.save(sess, 'sessModel/sessionSave.ckpt')



    















    
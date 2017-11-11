'''
Created on Nov 8, 2017

@author: dabrown

this is the main function with two CNN and two fc layers going out
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
from datetime import timedelta
import MainDir

npzfile = np.load(MainDir.dirData + 'saved_data_file33.npz')
train_segs = npzfile['train_segs']
train_labels = npzfile['train_labels']
train_labels_1h = npzfile['train_labels_1h']
test_segs = npzfile['test_segs']
test_labels = npzfile['test_labels']
test_labels_1h = npzfile['test_labels_1h']
val_segs = npzfile['val_segs']
val_labels = npzfile['val_labels']
val_labels_1h = npzfile['val_labels_1h']

num_iterations = 10000

# Convolutional Layer 1.
filter_size1 = 20          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 20         # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc1_size = 256             # Number of neurons in fully-connected layer.

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

# Number of classes, one class for each of 10 digits.
num_classes = 33

# define epsilon!
epsilon = 0.0018

# track iterations throughout run
total_iterations = 1

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
    layer = tf.nn.relu(layer)

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
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

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

def getMini_batch(x,y, batch_size):
    length = x.shape[0]
    
    xMini = np.empty([1,1,500,1])
    yMini = np.ndarray([1,num_classes])
    
    
    # stupid randint includes last value....
    indx = random.randint(0,length - 1)
    
    xsub = np.expand_dims(x[indx], axis = 0)
    ysub = np.expand_dims(y[indx], axis = 0)
    xMini = xsub
    yMini = ysub
    
    for i in range(1,batch_size):
        indx = random.randint(0,length-1)
        xsub = np.expand_dims(x[indx], axis = 0)
        ysub = np.expand_dims(y[indx], axis = 0)
        xMini = np.vstack((xMini,xsub))
        yMini = np.vstack((yMini,ysub))
    
    return xMini, yMini


# Counter for total number of iterations performed so far.


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations + 1):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = getMini_batch(train_segs, train_labels_1h, train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        sess.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = sess.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Split the test-set into smaller batches of this size.


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(test_segs)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0
    
    # do voting for a set of speaker data
    for speaker in range(num_classes):
        speaker_indx = test_labels == speaker
        
        x_segs = test_segs[speaker_indx]
        labels = test_labels_1h[speaker_indx]
        
        feed_dict = {x:x_segs, y_true:labels}
        
        cls_pred[speaker_indx] = sess.run(y_pred_cls, feed_dict=feed_dict)
        
        max_indx = 0
        max_sum = 0
        for indx in range(num_classes):
            sum = np.sum(cls_pred[speaker_indx] == indx)
            if(sum > max_sum):
                max_indx = indx
                max_sum = sum
                
        cls_pred[speaker_indx] = max_indx
        
#         
# 
#     while i < num_test:
#         # The ending index for the next batch is denoted j.
#         j = min(i + test_batch_size, num_test)
# 
#         # Get the images from the test-set between index i and j.
#         images = test_segs[i:j]
# 
#         # Get the associated labels.
#         labels = test_labels_1h[i:j]
# 
#         # Create a feed-dict with these images and labels.
#         feed_dict = {x: images,
#                      y_true: labels}
# 
#         # Calculate the predicted class using TensorFlow.
#         cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
# 
#         # Set the start-index for the next batch to the
#         # end-index of the current batch.
#         i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = test_labels

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def print_valid_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(val_segs)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0
    
    # do voting for a set of speaker data
    for speaker in range(num_classes):
        speaker_indx = val_labels == speaker
        
        x_segs = val_segs[speaker_indx]
        labels = val_labels_1h[speaker_indx]
        
        feed_dict = {x:x_segs, y_true:labels}
        
        cls_pred[speaker_indx] = sess.run(y_pred_cls, feed_dict=feed_dict)
        
        max_indx = 0
        max_sum = 0
        for indx in range(num_classes):
            sum = np.sum(cls_pred[speaker_indx] == indx)
            if(sum > max_sum):
                max_indx = indx
                max_sum = sum
                
        cls_pred[speaker_indx] = max_indx

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = test_labels

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on valid-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

if __name__ == '__main__':
    pass

    x = tf.placeholder(tf.float32, shape=[None, 1, img_size_flat, 1], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    
    
    layer_conv1, weights_conv1 = \
    new_conv_layer(input=x,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
    
    layer_conv1 = tf.nn.dropout(layer_conv1, keep_prob= 0.75)
    
    layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
    
    layer_flat, num_features = flatten_layer(layer_conv2)
    
    layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc1_size,
                         use_relu=True)
    
    layer_fc1 = tf.nn.dropout(layer_fc1, keep_prob=0.75)
    
    
    layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc1_size,
                         num_outputs=num_classes,
                         use_relu=False)
    
    
    y_pred = tf.nn.softmax(layer_fc2)
    
    y_pred_cls = tf.argmax(y_pred, axis=1)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
    
    cost = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=epsilon).minimize(cost)   
    
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
    
        optimize(num_iterations) # We already performed 1 iteration above.
    
        print_valid_accuracy()
        print_test_accuracy()
    
        # save output to commputer
        saver.save(sess, 'sessModel/sessionSave.ckpt')
















    
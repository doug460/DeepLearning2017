'''
Created on Aug 25, 2017

@author: dabrown
'''

import tensorflow as tf


if __name__ == '__main__':
    pass

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
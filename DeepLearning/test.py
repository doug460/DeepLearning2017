'''
Created on Aug 25, 2017

@author: dabrown
'''

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    pass

    array = np.array([[1,2,3]])
    array = np.repeat(array,2, axis = 0)
    print(array)
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:30:24 2017

@author: hp
"""

import numpy as np
import random
import datetime

def log(*context):
    '''
    for output some infomation
    '''
    outputlogo = "---->" + "[" + str(datetime.datetime.now()) + "]"
    string_print = ""
    for c in context:
        string_print += str(c)+"  "
    content = outputlogo +string_print + '\n'
    f = open("log.txt",'a')
    f.write(content)
    f.close;
    print outputlogo,string_print,'\n'
    return True

#loadData
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data(f):
    f = gzip.open( f , 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper( f ):
    tr_d, va_d, te_d = load_data(f)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

dataFile = './data/mnist.pkl.gz'
train_data , val_data , test_data = load_data_wrapper(dataFile)

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z) )

class Network(object):
    def __init__(self,sizes):
        '''
        size : network [3,2,1]
        biases weights are model parameters
        '''
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x , y in zip(sizes[:-1],sizes[1:])]
        
    def feedforward(self,a):
        for w , b in zip(self.weights , self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self , training_data  , epochs , mini_batch_size , eta , test_data = None):
        '''
        training_data : (input,output) 
        '''
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch , eta)
            if test_data:
                log("Epoch : %d , %d / %d" % ( j , self.evaluate(test_data) ,n_test))
            else:
                log("Epoch : %d fineshed" % j)
         
    def update_mini_batch(self , mini_batch , eta):
        '''
        grandient desent
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b , delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb , dnb in zip(nabla_b , delta_nabla_b)]
            nabla_w = [nw+dnw for nw , dnw in zip(nabla_w , delta_nabla_w)]
            self.weights = [w - (eta*nw)/len(mini_batch) for w , nw in zip(self.weights , nabla_w)]
            self.biases = [b - (eta*nb)/len(mini_batch) for b , nb in zip(self.biases , nabla_b)]
            
        
    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b , w in zip(self.biases , self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derviative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta , activations[-2].transpose())
        for l in xrange(2 , self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derviative(self , output_activations , y):
        return output_activations - y
            
    
    
net = Network([784,30,10])
net.SGD(train_data,30,10,3.0,test_data)
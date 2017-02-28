#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:27:17 2017

@author: hp
"""

import numpy as np
import random
import datetime

def Log(*context):
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

class CrossEntropyCost(object):
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z,a,y):
        return a-y
    
class QuadraticCost(object):
    @staticmethod
    def fn(a , y):
        return 0.5 * np.linalg.norm(a - y)**2
    
    @staticmethod
    def delta(z , a , y):
        return (a - y) * sigmoid_prime(z)
    
class Network(object):
    def __init__(self , sizes , cost = CrossEntropyCost ):
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y , 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y , x) /np.sqrt(x) for x,y in zip(self.sizes[:-1] , self.sizes[1:])]
        
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y , 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y , x ) for x , y in zip(self.sizes[:-1] , self.sizes[1:])]
        
    def feedforward(self , a):
        for b , w in zip(self.biases , self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def updata_mini_bach(self , mini_batch , eta , lmbda , n):
        nabla_b = [np.zeros(np.shape(b)) for b in self.biases]
        nabla_w = [np.zeros(np.shape(w)) for w in self.weights]
        for x , y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb , dnb in zip(nabla_b , delta_nabla_b)]
            nabla_w = [nw+dnw for nw , dnw in zip(nabla_w , delta_nabla_w)]
            
        self.weights = [ (1-(eta*lmbda/n))*w-eta/len(mini_batch)*nw 
                        for w , nw in zip(self.weights , nabla_w)]
        self.biases = [b - eta/len(mini_batch)*nb for b , nb in zip(self.biases , nabla_b)]
        
    def backprop(self , x , y):
        nabla_b = [np.zeros(np.shape(b)) for b in self.biases]
        nabla_w = [np.zeros(np.shape(w)) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = []
        for b , w in zip(self.biases , self.weights):
            z = np.dot(w , activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost.delta(zs[-1] , activations[-1] , y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta , activations[-2].transpose())
        for l in xrange(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose() , delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta , activations[-l-1].transpose())
        return (nabla_b , nabla_w)
    
    def accuracy(self , data , convert = False):
        if convert:
            results = [(np.argmax(self.feedforward(x)) , np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)) , y) for (x,y) in data]
        try:
            returnResult = sum(int(x == y) for x,y in results)
        except:
            import time
            for x,y in results:
                print convert
                print x
                print y
                print x==y
                time.sleep(10)
            raise 1
        return returnResult
    
    def total_cost(self , data , lmbda , convert = False):
        cost = 0.0
        for x , y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a , y) /len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def SGD(self , training_data , epochs , mini_batch_size , eta ,
            lmbda = 0.0 , evaluation_data=None, monitor_evaluation_cost=False, 
            monitor_evaluation_accuracy=False, monitor_training_cost=False, 
            monitor_training_accuracy=False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0 , n , mini_batch_size)]
            for mini_batch in mini_batches:
                self.updata_mini_bach(mini_batch , eta , lmbda , len(training_data))
            Log("Epoch %s training complete" % j)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data , lmbda ,True)
                evaluation_cost.append(cost)
                Log("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data )
                evaluation_accuracy.append(accuracy)
                Log("Accuracy on evaluation data: {} / {}".format(accuracy , n_data))
            if monitor_training_cost:
                cost = self.total_cost(training_data , lmbda)
                training_cost.append(cost)
                Log("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data,True)
                training_accuracy.append(accuracy)
                Log("Accuracy on training data: {} / {}".format(accuracy , n))
        return (evaluation_cost, evaluation_accuracy , training_cost, training_accuracy)
            
            
            
dataFile = './data/mnist.pkl.gz'
train_data , val_data , test_data = load_data_wrapper(dataFile)
net = Network([784,30,10])
evaluation_cost, evaluation_accuracy , training_cost, training_accuracy = net.SGD(train_data,epochs = 30, mini_batch_size = 10,eta = 3.0, lmbda = 0.0 , evaluation_data = test_data ,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, 
            monitor_training_accuracy=True)

import pygal
chart = pygal.Line(height=350)
chart.x_labels = map(str, range(0, 30))
chart.add('evaluation_cost' , evaluation_cost)
chart.add('training_cost' , training_cost)
chart.render_in_browser()

chart1 = pygal.Line(height=350)
chart1.x_labels = map(str, range(0, 30))
chart1.add('evaluation_accuracy' , evaluation_accuracy)
chart1.render_in_browser()

chart2 = pygal.Line(height=350)
chart2.x_labels = map(str, range(0, 30))
chart2.add('training_accuracy' , training_accuracy)
chart2.render_in_browser()
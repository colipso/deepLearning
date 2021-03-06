#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:09:04 2017

@author: hp
"""

import theano
import numpy as np
import cPickle
import gzip
import theano
import theano.tensor as T
from theano.tensor.nnet import conv , softmax , sigmoid
from theano.tensor import shared_randomstreams , tanh
from theano.tensor.signal import downsample
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

def linear(z):
    return z

def ReLU(z):
    return T.maximum(0.0 , z)

def dropout_layer(layer ,  p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n = 1 , p = 1-p_dropout , size = layer.shape)
    return layer * T.cast(mask , theano.config.floatX)

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

GPU = False
if GPU:
    Log("Run RNN model on GPU")
    try:
        theano.config.device = 'gpu'
    except:
        pass
    theano.config.floatX = 'float32'
else:
    Log("Run RNN model on CPU")
    try:
        theano.config.device = 'cpu'
    except:
        pass
    
    
dataFile = './data/mnist.pkl.gz'
def load_data_shared(filePath):
    f = gzip.open(filePath , 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        shared_x = theano.shared(np.asarray(data[0] , dtype = theano.config.floatX) , borrow = True)
        shared_y = theano.shared(np.asarray(data[1] , dtype = theano.config.floatX) , borrow = True)
        return shared_x , T.cast(shared_y , 'int32')
    return [shared(training_data) , shared(validation_data) , shared(test_data)]

### define layer types
class ConvPoolLayer(object):
    def __init__(self , filter_shape, image_shape, 
                 poolsize=(2, 2), activation_fn=sigmoid):
        """
        Example:
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                filter_shape=(20, 1, 5, 5),
                poolsize=(2, 2),
                activation_fn=ReLU)
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        n_out = filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize)
        self.w = theano.shared( 
                np.asarray(np.random.normal(loc = 0 , scale = np.sqrt(1.0 / n_out) , size = filter_shape) , dtype = theano.config.floatX) , 
                          borrow = True)
        self.b = theano.shared( 
                np.asarray(np.random.normal(loc = 0 , scale = 1 , size = filter_shape[0]) , dtype = theano.config.floatX) , 
                          borrow = True)
        self.params = [self.w , self.b]
        
    def set_input(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(input = self.inpt , filters = self.w , filter_shape = self.filter_shape ,  image_shape = self.image_shape)
        pooled_out = downsample.max_pool_2d(input = conv_out , ds = self.poolsize , ignore_border = True)
        self.output = self.activation_fn(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output
        
class FullyConnectedLayer(object):
    def __init__(self , n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        """
        Example
        --------
            FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5)
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.w = theano.shared(
                    np.asarray(
                            np.random.normal(loc = 0.0 , scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),dtype = theano.config.floatX
                                            ) , 
                    name='w', borrow=True
                )
        self.b = theano.shared(
                    np.asarray(
                            np.random.normal(loc = 0.0 , scale=1, size=(n_out,)),dtype = theano.config.floatX
                                            ) , 
                    name='b', borrow=True
                )
        self.params = [self.w , self.b]
        
    def set_input(self , inpt , inpt_dropout , mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size , self.n_in))
        self.output = self.activation_fn((1-self.p_dropout) * T.dot(self.inpt , self.w) + self.b)
        self.y_out = T.argmax(self.output , axis = 1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)) , self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout , self.w) + self.b)
        
    def accuracy(self , y):
        return T.mean(T.eq(self.y_out,y))
    
class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.w = theano.shared(np.zeros((self.n_in , self.n_out),dtype = theano.config.floatX) ,name='w', borrow=True)
        self.b = theano.shared(np.zeros((self.n_out ,),dtype = theano.config.floatX) ,name='b', borrow=True)
        self.params = [self.w , self.b]
        
    def set_input(self , inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size , self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w ) + self.b)
        self.y_out = T.argmax(self.output,axis =1)
        self.inpt_dropout = dropout_layer(
        inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)
        
    def cost(self ,net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
    
    def accuracy(self ,y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))



class Network(object):
    def __init__(self , layers , mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        
        Example
        --------
        Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                filter_shape=(20, 1, 5, 5),
                poolsize=(2, 2),
                activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                filter_shape=(40, 20, 5, 5),
                poolsize=(2, 2),
                activation_fn=ReLU),
                FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
                mini_batch_size)
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        init_layer = self.layers[0]
        init_layer.set_input(self.x , self.x ,self.mini_batch_size)
        for j in xrange(1 , len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_input(prev_layer.output , prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data, lmbda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size
                               
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_training_batches
                          
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)for param, grad in zip(self.params, grads)]
        
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
                                    [i], cost, updates=updates,
                                    givens={
                                    self.x:
                                    training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                                    self.y:
                                    training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                                    })
        validate_mb_accuracy = theano.function(
                                                [i], self.layers[-1].accuracy(self.y),
                                                givens={
                                                self.x:
                                                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                                                self.y:
                                                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                                                })
        test_mb_accuracy = theano.function(
                                            [i], self.layers[-1].accuracy(self.y),
                                            givens={
                                            self.x:
                                            test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                                            self.y:
                                            test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                                            })
        self.test_mb_predictions = theano.function(
                                                    [i], self.layers[-1].y_out,
                                                    givens={
                                                    self.x:
                                                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                                                    })
    
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 10000 == 0:
                    Log("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                    [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    Log("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        Log("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean([test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            Log('The corresponding test accuracy is {0:.2%}'.format(test_accuracy))
                            
        Log("Finished training network.")
        Log("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        Log("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        
        
training_data, validation_data, test_data = load_data_shared(dataFile)
mini_batch_size = 10
net = Network(  [
                ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                filter_shape=(20, 1, 5, 5),
                poolsize=(2, 2),
                activation_fn=ReLU),
                ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                filter_shape=(40, 20, 5, 5),
                poolsize=(2, 2),
                activation_fn=ReLU),
                FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
                SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)
                ],
                mini_batch_size)
net.SGD(training_data, 40, mini_batch_size, 0.03,validation_data, test_data)
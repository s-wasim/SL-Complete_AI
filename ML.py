import numpy as np
import tensorflow as tf
from random import sample
import pandas as pd
import math
import threading
import psutil

def test_train_split(self, x:np.ndarray, y:np.ndarray, train_sample_frac:float=0.8):
    assert 0 < train_sample_frac < 1
    assert y.shape[0] == x.shape[0]
    assert type(x) == np.ndarray and type(y) == np.ndarray
        
    train_len = int(y.shape[0] * train_sample_frac)
        
    init_set = list(range(y.shape[0]))
    train_set = set(sample(init_set, train_len))
    test_set = set(init_set) - train_set

    train_x, train_y = x[list(train_set)], y[list(train_set)]
    test_x, test_y = x[list(test_set)], y[list(test_set)]
    return train_x, train_y, test_x, test_y

class Regression:
    def __init__(self, features=1, w=None, b=tf.Variable(tf.random.normal([1, 1], dtype=np.float64)), dtype=np.float64):
        if w is None:
            w = tf.Variable(tf.random.normal([features, 1], dtype=np.float64), dtype=np.float64)
        self.__features = features
        self.__w = w
        self.__b = b
        self.__defined = False
        self.__err_fun = None
        self.__err_name = ''
        self.__activation = None
        self.__activation_name = ''
        self.__m_adam, self.__v_adam = None, None
        self.__optimizer = None

    def _describe(self):
        if self.__defined:
            print(f'Features/Dimesions: {self.__features}')
            print('w Value: ', end = '\n\t')
            print(*self.__w.numpy(), sep='\n\t')
            print(f'b Value: {self.__b.numpy()[0]}')
            print(f'Error Function->Error: {self.__activation_name}->{self.__err_name}')
            print(f'Optimizer: {self.__optimizer}')
            return
        print('Not Defined')
    
    def define(self, err='', err_func=None, activation='linear', optimizer='_', activation_func=None):
        if self.__defined:
            return 'Already Defined'
        self.__optimizer = optimizer
        self.__defined = True
        match err:
            case 'BinaryCrossEntropy':
                def __LOG(y, y_hat):
                    epsilon = 1e-7
                    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
                    return tf.reduce_mean((-y*tf.math.log(y_hat)) - ((1-y)*tf.math.log(1-y_hat)))
                self.__err_fun = __LOG
            case 'MeanSquareError':
                def __MSE(y, y_hat):
                    return tf.math.sqrt(tf.reduce_mean(tf.square(y - y_hat)))
                self.__err_fun = __MSE
            case _:
                if err_func is None:
                    raise NotDefinedError
                self.__err_fun = err_func
        match activation:
            case 'linear':
                def __lin(w, x, b):
                    return tf.matmul(x, w) + b
                self.__activation = __lin
            case 'sigmoid':
                def __sigmoid(w, x, b):
                    return tf.sigmoid(tf.matmul(x, w) + b)
                self.__activation = __sigmoid
            case _:
                if activation is None:
                    raise NotDefinedError
                self.__activation = activation_func
        self.__activation_name = activation
        self.__err_name = err
                
        return f'Activation: {activation}, Error: {err}'

    def predict(self, x:np.ndarray):
        x = tf.constant(x, dtype=np.float64)
        return self.__activation(self.__w, x, self.__b)

    def get_params(self):
        return self.__w.numpy(), self.__b.numpy()

    def __adam_optimizer(self, gradients, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if self.__m_adam is None:
            self.__m_adam, self.__v_adam = 0, 0
        for i in range(params.shape[0]):
            g = gradients[i, 0]
            # Update biased first moment estimate
            self.__m_adam = beta1 * self.__m_adam + (1 - beta1) * g
            # Update biased second raw moment estimate
            self.__v_adam = beta2 * self.__v_adam + (1 - beta2) * g**2
            # Compute bias-corrected first moment estimate
            m_hat = (self.__m_adam / (1 - beta1**(i+1))).numpy()
            # Compute bias-corrected second raw moment estimate
            v_hat = self.__v_adam / (1 - beta2**(i+1))
            params[i, 0].assign(params[i, 0].numpy() - (lr * m_hat / (np.sqrt(v_hat) + epsilon)))
        return params
        

    def train(self, y:np.ndarray, x:np.ndarray, learning_rate:float, iters:int, regularize:float=0.):
        y = tf.constant(y, dtype=np.float64)
        x = tf.constant(x, dtype=np.float64)
        loss = []
        for i in range(1, iters+1):
            with tf.GradientTape() as tape:
                y_hat = self.__activation(self.__w, x, self.__b)
                # Prevents over-fitting
                add_term = ((2*regularize)/y_hat.shape[0]) * tf.reduce_mean(tf.math.square(self.__w))
                J = self.__err_fun(y, y_hat) + add_term
            dJ_dw, dJ_db = tape.gradient(J, [self.__w, self.__b])
            
            if J <= 0 or np.isnan(J):
                print('TRAINING HALTED!! (Maximum Accuracy Achieved)')
                return loss
            
            match self.__optimizer:
                case 'Adam':
                    self.__w = tf.Variable(self.__adam_optimizer(dJ_dw, self.__w, learning_rate))
                    self.__b = tf.Variable(self.__adam_optimizer(dJ_db, self.__b, learning_rate))
                case _:
                    self.__w = tf.Variable(self.__w - (dJ_dw * learning_rate))
                    self.__b = tf.Variable(self.__b - (dJ_db * learning_rate))
            J = J.numpy()
            loss.append(J)
            
            progress = (50 * i) // iters
            print(' ' * 150, end='\r')
            print(f'PROGRESS: {"_" * progress}{"-"*(50 - progress)} / ITERS={i} - LOSS={J} LR={learning_rate}', end='\r')
            
        print('TRAINING COMPLETE!! (Iteration limit reached)')
        return loss

    def test(self, y:np.ndarray, x:np.ndarray, score_func=None):
        predictions = self.predict(x)
        
        J = self.__err_fun(y, predictions)
        
        MAE = np.mean(np.abs(y - predictions))
        MSE = np.mean(np.square(y - predictions))
        RMSE = np.sqrt(MSE)

        if score_func == None:
            score = 1 - (np.sum(np.square(y - predictions))/np.sum(np.square(y - y.mean())))
        else:
            score = score_func(y, predictions)

        return '\n'.join([
            'Error evaluations:',
            f'\tAvg. Error from Error Function: {J:.4f}',
            f'\tMean Absolute Error: {MAE:.4f}',
            f'\tMean Squared Error: {MSE:.4f}',
            f'\tRoot Mean Squared Error: {RMSE:.4f}',
            f'\tAccuracy Score for Model (Total Defined_Score): {score*100:.4f}%'
        ])
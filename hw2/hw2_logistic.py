# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:22:42 2020

@author: User
"""

import numpy as np
import sys


X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
X_mean = np.load('mean.npy')     
X_std = np.load('std.npy')
w = np.load('w.npy')
b = np.load('b.npy')

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1-1e-8)

def get_prob(X, w, b):
    return _sigmoid(np.add(np.matmul(X, w), b))
def infer(X, w, b):
    return np.round(get_prob(X, w, b))

X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = None, X_mean=X_mean, X_std=X_std)

result = infer(X_test, w, b)

with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(result):      
        f.write('%d,%d\n' %(i, v))
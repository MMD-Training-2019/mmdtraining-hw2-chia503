# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:16:35 2020

@author: User
"""

import numpy as np
import sys

X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
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

def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc


X_train, X_mean, X_std = _normalize_column_normal(X_train, train = True)
X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = None, X_mean=X_mean, X_std=X_std)

X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)

size = X_train.shape[1]
cov_0 = np.zeros((size, size))
cov_1 = np.zeros((size, size))

for x in X_train_0:
    cov_0 = cov_0 + np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 = cov_1 + np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]
    
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

u, s, v = np.linalg.svd(cov, full_matrices = False)
inv = np.matmul(v.T * 1/s, u.T)

w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

Y_train_pred = 1 - infer(X_train, w, b)
print('accuracy:{}'.format(accuracy(Y_train_pred, Y_train)))

pred = 1 - infer(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(pred):
        f.write('{},{}\n'.format(i, label))
        
    
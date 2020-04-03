# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:17:04 2020

@author: User
"""

import numpy as np
import xgboost as xgb
import sys

X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
        
model = xgb.Booster({'nthread':4}) # init model
model.load_model('test.model')
#model = xgb.load_model('test.model')
dxtest = xgb.DMatrix(X_test)
pred = model.predict(dxtest)
result = [round(value) for value in pred]

with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(result):      
        f.write('%d,%d\n' %(i, v))


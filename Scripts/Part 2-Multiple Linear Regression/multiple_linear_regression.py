#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:41:31 2017

@author: pranshu
"""

import os
import numpy as np
import pandas as pd

path=os.path.pardir + '/data/ex1data2.txt'
data=pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
data_norm=(data-data.mean())/data.std()
data_norm.insert(0,'ones',1)

X=np.matrix(data_norm.iloc[:,:-1].values)
Y=np.matrix(data_norm.iloc[:,-1].values).T
theta=np.matrix(np.zeros((1,3),dtype=np.int64))

def compute_cost(X,Y,theta):
    inner=np.power((X*theta.T-Y),2)
    return np.sum(inner)/(2*len(X))

def gradient_descent(X,Y,theta,alpha,iters):
    for i in range(iters):
        error=X*theta.T-Y
        theta=theta-alpha*(error.T*X)/len(X)
        cost=compute_cost(X,Y,theta)
        
    return theta,cost

t,c=gradient_descent(X,Y,theta,0.01,1000)
print t,c
    
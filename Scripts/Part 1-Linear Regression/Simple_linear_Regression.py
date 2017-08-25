#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:35:48 2017

@author: pranshu
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path=os.path.pardir+ '/ipython-notebooks-master/data/ex1data1.txt'
data=pd.read_csv(path,names=['Population','Profit'])
data.insert(0,'ones',1)

X=np.matrix(data.iloc[:,:-1].values)
Y=np.matrix(data.iloc[:,-1].values).T
theta=np.matrix(np.zeros((1,2),dtype=np.int64))   
                
def get_cost(X,Y,theta):
    inner=np.power(((X*(theta.T))-Y),2)
    return np.sum(inner)/(2*len(X))

def gradientdescent(X,Y,theta,alpha,iters):
    for i in range(iters):
        error=X*(theta.T)-Y
        theta=theta - alpha*(error.T*X)/(len(X))
        #print "theta is {}\n".format(theta) 
        cost=get_cost(X,Y,theta)
        #print "cost is {}\n".format(cost)
    return theta,cost

print "theta is {}\n".format(theta)
cost=get_cost(X,Y,theta)
print "cost is {}\n".format(cost)
t,c=gradientdescent(X,Y,theta,0.01,1000)
print "cost is {}\n".format(c)
    
#plt.scatter(X,Y)
#plt.title('Profit vs Population')
#plt.xlabel('Population')
#plt.ylabel('Profit')
#plt.show()
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 03:26:51 2017

@author: pranshu
"""

#Logistic Regression

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path=os.path.pardir+'/data/ex2data2.txt'


data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
degree = 5  
x1 = data['Test 1']  
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

for i in range(1, degree):  
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('Test 1', axis=1, inplace=True)  
data.drop('Test 2', axis=1, inplace=True)

 

'''#separating admitted and non admitted candidates
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]
'''
#plotting the admitted and non admitted candidate's exam 1 vs exam 2 score
'''
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'],s=50, c='b', marker='o',label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'],s=50, c='r', marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')'''

#defining a sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


def cost(theta,X,Y,reg_par):
    theta=np.matrix(theta)
    X=np.matrix(X)
    Y=np.matrix(Y)
    first=np.multiply(-Y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-Y),np.log(1-sigmoid(X*theta.T)))
    reg=(reg_par/(2*len(X)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/(len(X)) + reg


cols=data.shape[1]
X=np.array(data.iloc[:,1:cols].values)
Y=np.array(data.iloc[:,0:1].values)
theta=np.zeros(11)
reg_par=1

print "cost before gradient descent {}".format(cost(theta,X,Y,reg_par))
#performing a single step of gradient descent. 
#We will use SciPy's optimization API 
#to optimize the parameters given functions to compute the cost and the gradients.
def gradient(theta,X,Y,reg_par):
    theta=np.matrix(theta)
    X=np.matrix(X)
    Y=np.matrix(Y)
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    
    error=sigmoid(X*theta.T)-Y
    
    for i in range(parameters):
        term=np.multiply(error,X[:,i]) #X[:,0] -> all rows and 0th column
        if(i==0):
            grad[i]=np.sum(term)/len(X)
        else:
            grad[i]=np.sum(term)/len(X) + ((reg_par/len(X))*theta[:,i])
    
    return grad

#SciPy's optimization API
import scipy.optimize as opt
result=opt.fmin_tnc(func=cost,x0=theta, fprime=gradient, args=(X,Y,reg_par))
print "\ncost after gradient descent {}".format(cost(result[0],X,Y,reg_par))

def predict(theta,X):
    prob=sigmoid(X*theta.T)
    return [1 if x>0.5 else 0 for x in prob]

theta_min=np.matrix(result[0])
predictions=predict(theta_min,X)
correct=[]
for a,b in zip(predictions,Y):
    if((a==1 and b==1) or (a==0 and b==0)):
        correct.append(1)
    else:
        correct.append(0)

accuracy=(sum(map(int,correct)))

            
    

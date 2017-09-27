#Logistic Regression

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path=os.path.pardir+'/data/ex2data1.txt'
data=pd.read_csv(path,header=None, names=['Exam 1','Exam 2','Admitted'])


 

#separating admitted and non admitted candidates
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]

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


def cost(theta,X,Y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    Y=np.matrix(Y)
    first=np.multiply(Y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-Y),np.log(1-sigmoid(X*theta.T)))
    return -np.sum(first+second)/(len(X))

data.insert(0,'Ones',1)

X=np.array(data.iloc[:,:-1].values)
Y=np.array(data.iloc[:,3:4].values)
theta=np.zeros(3)

print "cost before gradient descent {}".format(cost(theta,X,Y))
#performing a single step of gradient descent. 
#We will use SciPy's optimization API 
#to optimize the parameters given functions to compute the cost and the gradients.
def gradient(theta,X,Y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    Y=np.matrix(Y)
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    
    error=sigmoid(X*theta.T)-Y
    
    for i in range(parameters):
        term=np.multiply(error,X[:,i]) #X[:,0] -> all rows and 0th column
        grad[i]=np.sum(term)/len(X) #grad[i] is a scalar value
    
    return grad

#SciPy's optimization API
import scipy.optimize as opt
result=opt.fmin_tnc(func=cost,x0=theta, fprime=gradient, args=(X,Y))
print "\ncost after gradient descent {}".format(cost(result[0],X,Y))

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

            
    

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 00:46:57 2017

@author: kevin
"""

import numpy as np
from pandas import DataFrame
from mnist import MNIST

# define the sigmoid gate
def sigmoid(z,derivative=False):
    if derivative==True:
        return z*(1-z)
    return 1/(1+np.exp(-z))

def MLP(x,weights,biases):
    out=x.copy()
    for k in range(deepness+1):
        out=sigmoid(np.dot(weights[k],out)+biases[k])
    return out

# load the images and labels
mndata = MNIST("mnist-data")
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

#hyperparameters:
h_n=200 #number of neurons per hidden layer
deepness=2 #number of hidden layers
gamma=0.02 #learning rate
batchSize=100

#constants
numOfTrainImages=60000
numOfTestImages=10000
sizeOfImages=28*28

#synaptic weights
if deepness>0:
    weights=[np.random.random((h_n,sizeOfImages))-0.5]
    biases=[np.random.random(h_n)-0.5]
    for _ in range(deepness-1):
        weights.append(np.random.random((h_n,h_n))-0.5)
        biases.append(np.random.random(h_n)-0.5)
    weights.append(np.random.random((10,h_n))-0.5)
    biases.append(np.random.random(10)-0.5)
else:
    weights=[np.random.random((10,sizeOfImages))-0.5]
    biases=[np.random.random(10)-0.5]
    
trainLoss=0
testLoss=0
weightsGradients=[]
biasesGradients=[]
trainLosses=[]
testLosses=[]
weightsUpdates=[np.zeros_like(w) for w in weights]
biasesUpdates=[np.zeros_like(b) for b in biases]
iteration=0
while True:
    iteration+=1
    print('iteration:',iteration)
    for i in range(numOfTrainImages):
        #feedforward
        a=[np.array(trainImages[i])]
        for j in range(deepness+1):
            a.append(sigmoid(np.dot(weights[j],a[-1])+biases[j]))
        #loss    
        d=a[-1].copy()
        d[int(trainLabels[i])]+=-1
        trainLoss+=0.5*np.sum(d**2)/batchSize
        #backforward
        for j in reversed(range(deepness+1)):
            if j==deepness:
                dz=d*sigmoid(a[j+1],True)
                    
            else:
                dz=np.dot(dz,weights[j+1]*sigmoid(a[j+1],True))
            weightsGradients.append(np.outer(dz,a[j]))
            biasesGradients.append(dz)
        #acumulate the gradients in the batches
        for k in range(deepness+1):
            weightsUpdates[k]+=-gamma*weightsGradients[::-1][k]/batchSize
            biasesUpdates[k]+=-gamma*biasesGradients[::-1][k]/batchSize
        
        if (i+1)%batchSize==0:
            for k in range(deepness+1):
                weights[k]+=weightsUpdates[k]
                biases[k]+=biasesUpdates[k]
            weightsUpdates=[np.zeros_like(w) for w in weights]
            biasesUpdates=[np.zeros_like(b) for b in biases]
            weightsGradients=[]
            biasesGradients=[]
            print('train loss:',trainLoss)
            trainLosses.append(trainLoss)
            trainLoss=0
            for l in range(numOfTestImages):
                testOut=MLP(np.array(testImages[l]),weights,biases)
                testOut[int(testLabels[l])]+=-1
                testLoss=0.5*(testOut**2)/numOfTestImages
            testLosses.append(testLoss)
            print('test loss:', testLoss)
            
data={'Train Loss':trainLosses,'Test Loss':testLosses}
frame=DataFrame(data)
frame.plot()
                

    




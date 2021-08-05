# This is an simple algorithm for mnist digit images classification
# This algorithm is just a NN with simple SGD 
# Author: Kevin Juan Román Rafaele

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

# data loading #
print("Loading MNIST data...")
mndata = MNIST("mnist-data")
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()
print("Data downloaded")

# NN hyperparameters
TRAINING_SIZE = 60000
TEST_SIZE = 10000
INPUT_LENGTH = 28*28
HIDDEN_LENGTH = 1000
OUTPUT_LENGTH = 10

e = 0.0001 # learning rate

TRAINING_TIME = 10

# preparing targets #
trainT = np.zeros((TRAINING_SIZE,OUTPUT_LENGTH))
testT = np.zeros((TEST_SIZE,OUTPUT_LENGTH))
for n in range(TRAINING_SIZE):
    trainT[n][trainLabels[n]] = 1
for n in range(TEST_SIZE):
    testT[n][testLabels[n]] = 1

# building NN #
W1 = np.random.normal(size = (INPUT_LENGTH,HIDDEN_LENGTH)) / (INPUT_LENGTH*HIDDEN_LENGTH)
W2 = np.random.normal(size = (HIDDEN_LENGTH,OUTPUT_LENGTH)) / (HIDDEN_LENGTH*OUTPUT_LENGTH)

B1 = np.zeros(HIDDEN_LENGTH)
B2 = np.zeros(OUTPUT_LENGTH)

dW1 = np.zeros_like(W1)
dW2 = np.zeros_like(W2)

dB1 = np.zeros_like(B1) 
dB2 = np.zeros_like(B2)

L0 = np.array(trainImages)
L1 = np.zeros((TRAINING_SIZE,HIDDEN_LENGTH))
L2 = np.zeros((TRAINING_SIZE,OUTPUT_LENGTH))
Y = np.zeros_like(L2)

dL1 = np.zeros_like(L1)
dL2 = np.zeros_like(L2)

# preparing plots #
fig, axs = plt.subplots(2,5)
nnComps = np.zeros((28,28,OUTPUT_LENGTH))
recLoss = np.zeros(TRAINING_TIME)

# softmax function #
def softmax(z):
    zmax = np.amax(z)
    num = np.exp(z-zmax)
    denom = np.sum(num,axis=1)
    return num / denom.reshape((len(denom),1))

def genTestLoss():
    p = softmax(np.dot(np.dot(testImages,W1)+B1,W2)+B2)
    if(mean_square): return np.mean(np.sum((p-testT)**2,axis=1))
    if(cross_entropy): return np.mean(np.sum(-testT*np.log(p),axis=1))

# run #
print("Start training")
for t in range(TRAINING_TIME):
    # feedforward
    L1 = np.dot(L0,W1) + B1
    L2 = np.dot(L1,W2) + B2
    Y = softmax(L2)
    
    # backprop #
    dL2 = (Y - trainT)
    dW2 = np.dot(L1.T,dL2) / TRAINING_SIZE
    dB2 = np.sum(dL2,axis=0) / TRAINING_SIZE
    
    dL1 = np.dot(dL2,W2.T)
    dW1 = np.dot(L0.T,dL1) / TRAINING_SIZE
    dB1 = np.sum(dL1,axis=0) / TRAINING_SIZE
    
    W1 -= e * dW1
    W2 -= e * dW2
    
    B1 -= e * dB1
    B2 -= e * dB2
    
    # calculating loss #
    train_loss = np.mean(np.sum(-trainT * np.log(Y),axis=1))
    recLoss[t] = train_loss
    # printing loss
    printLoss = "Train Loss = "+str(train_loss)+" -- Test Loss = "+str(genTestLoss())
    print(printLoss)
    
    # ploting #
    if( t%2 == 0 ):
        nnComps = np.dot(W1,W2)
        for a in range(2):
            for b in range(5):
                i = 2*a + b
                axs[a,b].imshow(nnComps[:,i].reshape((28,28)),cmap='gray',label="C["+str(i)+"]")
                axs[a,b].axis('off')
        
        handles, labels = axs[0,4].get_legend_handles_labels()
        fig.legend(handles, labels,title='Time = '+str(t)) 
        plt.pause(0.0000000001)
        fig.savefig("genComps.png") # saving figure
        for a in range(2):
            for b in range(5):
                axs[a,b].cla()

        
    

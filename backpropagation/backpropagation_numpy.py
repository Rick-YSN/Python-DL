import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time



def sigmoid(x):
    '''
    Sigmoid 函数
    :param x:
    :return:
    '''
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    '''
    Sigmoid 导函数
    :param x:
    :return:
    '''
    m = sigmoid(x)
    return m*(1.0-m)

def softmax(x):
    return (np.exp(np.array(x)) / np.sum(np.exp(np.array(x))))

def softmax_prime(x):
    return softmax(x)*(1.0-softmax(x))

INPUTS = 784
HIDDEN = 30
OUTPUTS = 10
EPOCHS = 2000
BATCH_SIZE = 1000
LEARNING_RATE = 0.01


# Timer start
start_time = time.time()

w1 = np.zeros((INPUTS, HIDDEN),dtype=np.float32)
b1 = np.zeros(HIDDEN,dtype=np.float32)
w2 = np.zeros((HIDDEN, OUTPUTS),dtype=np.float32)
b2 = np.zeros(OUTPUTS,dtype=np.float32)

mnist = input_data.read_data_sets('./data/', one_hot=True)

# Epochs loop
for k in range(EPOCHS):

    x, y = mnist.train.next_batch(BATCH_SIZE)

    # Hidden layer activation
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    # Outputs
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    quadractic_cost =np.subtract(a2, y)

    # Deltas
    dz2 = np.multiply(quadractic_cost, softmax_prime(z2))
    db2 = np.mean(dz2, axis =0,keepdims=False)
    dz1 = np.multiply(sigmoid_prime(z1), np.dot(dz2, np.transpose(w2)))

    # Weights update
    dw2 =  np.multiply(LEARNING_RATE, np.dot(np.transpose(a1),dz2))
    dw1 = np.multiply(LEARNING_RATE, np.dot(np.transpose(x),dz1))
    db1 = np.mean(dz1,axis=0,keepdims=False)

    # Weights update
    w1 -= LEARNING_RATE * dw1
    b1 -= LEARNING_RATE * db1
    w2 -= LEARNING_RATE * dw2
    b2 -= LEARNING_RATE * db2

    accuracy_mat = np.equal(np.argmax(a2, 1), np.argmax(y, 1))
    accuracy_mat = accuracy_mat.astype(np.float32)
    accuracy_result = np.mean(accuracy_mat)

    # Print training status
    print('EPOCH: {0:4d}/{1:4d}\t\tElapse time [seconds]: {2:5f}\t\tAccuracy: {3:5f}\n'.format(k, EPOCHS, time.time() - start_time, accuracy_result))




import numpy as np
import pandas as pd

np.random.seed(42)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
def prediction(X, W, b):
    return sigmoid(np.matmul(X,W)+b)
def error_vector(y, y_hat):
    return [-y[i]*np.log(y_hat[i]) - (1-y[i])*np.log(1-y_hat[i]) for i in range(len(y))]
def error(y, y_hat):
    ev = error_vector(y, y_hat)
    return sum(ev)/len(ev)


def softmax(L):
    '''
    :param L: list of numbers
    :return: list of values given by the softmax function
    '''
    list = []
    for score in L:
        value = np.exp(score)/sum(np.exp(L))
        list.append(value)
    return list

def cross_entropy(Y, P):
    '''
    :param Y: list of Y
    :param P: list of P
    :return: float corresponding to Y and P cross-entropy
    '''
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


def dErrors(X, y, y_hat):
    '''
    :param X: list of values
    :param y: list of labels
    :param y_hat: prediction
    :return: gradients (partial derivatives) with respect to each values and the biais
    '''
    # The first list contains the gradient (partial derivatives) with respect to w1
    DErrorsDx1 = [-X[i][0]*(y[i]-y_hat[i]) for i in range(len(y))]
    # The second list contains the gradient (partial derivatives) with respect to w2
    DErrorsDx2 = [-X[i][1]*(y[i]-y_hat[i]) for i in range(len(y))]
    # The third list contains the gradient (partial derivatives) with respect to b
    DErrorsDb = [-(y[i]-y_hat[i]) for i in range(len(y))]
    return DErrorsDx1, DErrorsDx2, DErrorsDb


def gradientDescentStep(X, y, W, b, learn_rate = 0.01):
    '''
    Implement the gradient descent step.
    Calculate the prediction, the gradients, and use them to update the weights and bias W, b. Then return
    the updated W and b.
    The error e will be calculated and returned for plotting purposes.
    :param X: list of values
    :param y: list of labels
    :param W: array of weights
    :param b: biais
    :param learn_rate: learning rate
    :return:
    '''
    y_hat = prediction(X,W,b)
    errors = error_vector(y, y_hat)
    derivErrors = dErrors(X, y, y_hat)
    W[0] -= sum(derivErrors[0])*learn_rate
    W[1] -= sum(derivErrors[1])*learn_rate
    b -= sum(derivErrors[2])*learn_rate
    return W, b, sum(errors)


def trainLR(X, y, learn_rate = 0.01, num_epochs = 100):
    '''
    This function runs the perceptron algorithm repeatedly on the dataset
    :param X: list of values
    :param y: list of labels
    :param learn_rate: learning rate
    :param num_epochs: number of epochs (iteration)
    :return:  few of the boundary lines obtained in the iterations, for plotting purposes.
    '''
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    # Initialize the weights randomly
    W = np.array(np.random.rand(2,1))*2 -1
    b = np.random.rand(1)[0]*2 - 1
    # These are the solution lines that get plotted below.
    boundary_lines = []
    errors = []
    for i in range(num_epochs):
        # In each epoch, we apply the gradient descent step.
        W, b, error = gradientDescentStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        errors.append(error)
    return boundary_lines, errors

data = pd.read_csv('data.csv')

data.columns = ['x1', 'x2', 'label']
values = np.array(data[["x1","x2"]])
labels = np.array(data[["label"]])

print trainLR(values, labels)

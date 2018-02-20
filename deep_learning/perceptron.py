import numpy as np
import pandas as pd

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    '''
    :param X: data
    :param y: labels
    :param W: weights (array)
    :param b: bias
    :param learn_rate: learning_rate
    :return: updated weigths and bias
    '''
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i] - y_hat == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    '''
    # This function runs the perceptron algorithm repeatedly on the dataset,
    # and returns a few of the boundary lines obtained in the iterations,
    # for plotting purposes.
    :param X: data
    :param y: labels
    :param learn_rate: learning rate
    :param num_epochs: number of epochs (iteration)
    :return: boundary lines
    '''
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])

    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines


data = pd.read_csv('data.csv')

data.columns = ['x1', 'x2', 'label']
values = np.array(data[["x1","x2"]])
labels = np.array(data[["label"]])

print trainPerceptronAlgorithm(values, labels)

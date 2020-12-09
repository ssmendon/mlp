from collections import defaultdict
import sys

from mendon.nn import Perceptron, Network

import numpy as np
import numpy.matlib
import matplotlib.pyplot as pyplot


def main():

    # load the data
    class1, labels1 = load_data('./data/Train1.txt', 0)
    class2, labels2 = load_data('./data/Train2.txt', 1)

    # normalization
    normalized_data = normalize(
        np.vstack([class1[:1500,:], class2[:1500,:]]).transpose(),
        np.vstack([class1, class2]).transpose(),
    )

    # separate train and validation
    trainX = np.hstack([
        normalized_data[:,:1500], normalized_data[:,2000:3500]
    ])
    trainY = np.vstack([np.zeros((1500,1)), np.ones((1500,1))])

    testX = np.hstack([
        normalized_data[:,1500:2000], normalized_data[:,3500:]
    ])
    testY = np.vstack([np.zeros((500,1)), np.ones((500, 1))])

    # train
    hidden_sizes = [2, 4, 6, 8, 10]
    models = defaultdict(lambda: [])
    for size in hidden_sizes:
        for i in range(10):
            print('Training with size {} on iter {}'.format(size, i))

            n = Network(2, size, 1)
            train(trainX, trainY, testX, testY, 0.002, n)

            models[size].append(n)

            # save a contour plot of the decision boundary
            plot_decision_boundary(
                './img/size-{}-iter-{}.jpg'.format(size, i), 
                normalized_data, 
                np.vstack([np.zeros((2000,1)), np.ones((2000, 1))]).astype(int), 
                n
            )
        
        print('\n')
    
    # evaluate performance of models

def plot_decision_boundary(file: str, data: np.ndarray, labels: np.ndarray, n: Network):
    r'''Plots the decision region given a trained model and labelled data.

    This is adapted from my Project 1 code, which in turn was adapted from:
    https://hackernoon.com/how-to-plot-a-decision-boundary-for-machine-learning-algorithms-in-python-3o1n3w07
    
    Args:
        file: path where the file is saved (directories should already exist)
        data: a column-vector (feat x samples)
        labels: a column vector (samples x 1)
        n: the trained network which accepts column vectors and outputs a row vector prediction
    '''

    # get the lower and upper bounds of the data
    lower = data.min(axis=1) - 1
    upper = data.max(axis=1) + 1
    
    # move to an integer which widens the region as much as possible
    lower, upper = np.floor(lower).astype(int), np.ceil(upper).astype(int)

    xmin, xmax = lower[0], upper[0]
    ymin, ymax = lower[1], upper[1]

    # grid between min and max in steps of 0.1
    xgrid = np.arange(xmin, xmax, 0.1)
    ygrid = np.arange(ymin, ymax, 0.1)

    # turn it into a grid of x, y points that span the space
    xx, yy = np.meshgrid(xgrid, ygrid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # stack them horizontally to obtain the coordinates x, y as row vectors
    # make predictions for each
    grid = np.hstack((r1, r2))
    predictions = n(grid.T).round().T

    # make the predictions the same shape as xx for contourf
    # contourf chooses a pair of indices i,j from xx and yy and checks index zz(i, j) = xx(i, j), yy(i, j)
    zz = predictions.reshape(xx.shape)

    fig = pyplot.figure(figsize=(15,10))
    pyplot.contourf(xx, yy, zz, cmap='Paired')  # decision boundary

    # draw the points in data colored by their class
    xy0 = data[:, np.nonzero(labels.flatten() == 0)]
    xy0.reshape((2, xy0.shape[-1]))
    xy1 = data[:, np.nonzero(labels.flatten() == 1)]
    xy1.reshape((2, xy1.shape[-1]))
    
    
    pyplot.scatter(*xy0, label='0', cmap='Paired')
    pyplot.scatter(*xy1, label='1', cmap='Paired')

    # get accuracy over all of data (train + valid)
    predict = n(data).transpose()
    accuracy = predict.round() == labels
    accuracy = np.mean(accuracy)

    pyplot.title('train accuracy {:.4%}'.format(accuracy))
    pyplot.legend()
    pyplot.show()
    fig.savefig(file)
    pyplot.close()

def normalize(train: np.ndarray, combined: np.ndarray) -> np.ndarray:

    # mean and std of each feature
    # each row is one feature
    mean = np.mean(train, axis=1)
    std = np.std(train, axis=1, ddof=1)

    # turn into column vectors from 1d
    mean = mean[..., np.newaxis]
    std = std[..., np.newaxis]

    return (combined - mean)/std

def load_data(file: str, label: int) -> (np.ndarray, np.ndarray):
    data = np.loadtxt(file)
    num_samples = data.shape[0]
    labels = np.ones((num_samples,1)) if label else np.zeros((num_samples, 1))

    return data, labels

def train(trainX, trainY, testX, testY, eta, n):

    n_samples = trainX.shape[1]
    epochs = 0

    prev_err = n.error(n(testX).T, testY)/n_samples
    epsilon = 0.000002

    done = False
    while not done:
        # randomly select some order
        order = list(range(n_samples))
        np.random.shuffle(order)

        # single sample training
        # pick a random number of samples to individually run and update
        
        for j in order:
            x = trainX[:, j]
            x = x[..., np.newaxis]
            t = trainY[j]
            n.backward_online(x, t, eta)

        epochs += 1

        # stopping condition
        # error not decreasing much anymore
        # could use np.linalg.norm(n.error_deriv) < epsilon, using tangent line approximation instead
        fwd = n(testX).transpose()
        err = n.error(fwd, testY)/n_samples
        if np.abs(err - prev_err) <= epsilon:
            done = True  # just to print accuracy one final time
        else:
            prev_err = err

        # accuracy
        if not epochs % 10 or done:
            predict_train = n(trainX).transpose()
            predict_validation = fwd

            train_accuracy = predict_train.round() == trainY
            train_accuracy = np.mean(train_accuracy)
            validation_accuracy = predict_validation.round() == testY
            validation_accuracy = np.mean(validation_accuracy)

            prompt = 'Training set accuracy after {} epochs:'.format(epochs).ljust(45)
            print(prompt + '{:.4%}'.format(train_accuracy))

            prompt = 'Validation set accuracy after {} epochs:'.format(epochs).ljust(45)
            print(prompt + '{:.4%}'.format(validation_accuracy))
            print()

        # kill condition
        # learning rate is too high and we're oscillating
        if epochs >= 300:
            done = True


if __name__ == '__main__':
    main()
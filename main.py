from collections import defaultdict
import sys

from mendon.nn import Perceptron, Network

import numpy as np
import numpy.matlib
import matplotlib.pyplot as pyplot

pyplot.rcParams['figure.figsize'] = [7.4, 5.8]
pyplot.rcParams['figure.dpi'] = 150

def main():

    # load the data
    class1 = np.loadtxt('./data/Train1.txt')
    class2 = np.loadtxt('./data/Train2.txt')

    test1 = np.loadtxt('./data/Test1.txt')
    test2 = np.loadtxt('./data/Test2.txt')

    # normalization
    normalized_data = normalize(
        np.vstack([class1[:1500,:], class2[:1500,:]]).transpose(),
        np.vstack([class1, class2]).transpose(),
    )

    # normalize testing set
    testX = normalize(
        np.vstack([class1[:1500,:], class2[:1500,:]]).transpose(),
        np.vstack([test1, test2]).transpose(),
    )
    testY = np.vstack([np.zeros((1000, 1)), np.ones((1000, 1))])

    # separate train from validation
    trainX = np.hstack([
        normalized_data[:,:1500], normalized_data[:,2000:3500]
    ])
    trainY = np.vstack([np.zeros((1500,1)), np.ones((1500,1))])

    validationX = np.hstack([
        normalized_data[:,1500:2000], normalized_data[:,3500:]
    ])
    validationY = np.vstack([np.zeros((500,1)), np.ones((500, 1))])

    # train
    eta = 0.1 #eta = 0.002
    hidden_sizes = [2, 4, 6, 8, 10]
    models = defaultdict(list)
    for size in hidden_sizes:
        for i in range(10):  # run each model 10 times
            print('Training with size {} on iter {}'.format(size, i))

            n = Network(2, size, 1)
            losses = train(trainX, trainY, validationX, validationY, testX, testY, eta, n)

            models[size].append(n)

            # save a contour plot of the decision boundary for the training data
            # and the learning curves
            plot_decision_boundary(
                './img/decision-boundary/size-{}-iter-{}.jpg'.format(size, i), 
                normalized_data, 
                np.vstack([np.zeros((2000,1)), np.ones((2000, 1))]).astype(int), 
                n
            )
            plot_learning_curves(
                './img/learning-curves/size-{}-iter{}.jpg'.format(size, i),
                losses
            )
            
        
        print('\n')
    
    print('Evaluating model performance...')
    # evaluate performance of models
    # use testing set and take the average of all 10 models
    # also: take the best, for comparison
    avg_accuracies = []
    best_accuracies = []
    for size in hidden_sizes:
        nets = models[size]

        accuracies = np.array([np.mean(n(testX).T.round() == testY) for n in nets])
        avg_accuracies.append(np.mean(accuracies))
        best_accuracies.append(np.amax(accuracies))

    plot_accuracy('./img/accuracies/chart.jpg', hidden_sizes, avg_accuracies, best_accuracies)

    print('Size'.rjust(25), end='')
    print('Average'.rjust(20), end='')
    print('Best'.rjust(20))
    for i in range(len(hidden_sizes)):
        print(str(hidden_sizes[i]).rjust(25), end='')
        print('{:.4%}'.format(avg_accuracies[i]).rjust(20), end='')
        print('{:.4%}'.format(best_accuracies[i]).rjust(20))


def plot_accuracy(file: str, hidden_sizes: list[int], avg_accuracies: list[float], best_accuracies: list[float]) -> None:
    r'''Plots the accuracy vs. hidden layer size.

    Args:
        file: path to where the file is saved (directories should already exist)
        hidden_sizes: a list of what hidden sizes were used
        avg_accuracies: a list of the average accuracy for that hidden layer size
        best_accuracies: a list of the best accuracy for that hidden layer size
    '''
    
    fig, ax = pyplot.subplots()

    ax.plot(hidden_sizes, avg_accuracies, 'r--', label='Average Loss')
    ax.plot(hidden_sizes, best_accuracies, 'b--', label='Best Loss')

    ax.set(xlabel='hidden size', ylabel='accuracy (%)',
           title='Accuracy')

    ax.legend()

    fig.savefig(file)
    pyplot.close(fig)


def plot_learning_curves(file: str, losses: dict[str, list[int, ...]]) -> None:
    r'''Plots the learning curves for a given set of train, validation, and testing losses.

    The code is adapted from the matplotlib sample:
    https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/simple_plot.html

    Args:
        file: path where the file is saved (directories should already exist)
        losses: a dictionary with keys 'train', 'validation', and 'test' storing a list of loss/n_samples
            (starting at epoch = 1 until epoch = len(list))
    '''

    num_epochs = len(losses['train'])
    if not num_epochs:
        print('No epochs to plot')
    epochs_axis = np.arange(0, num_epochs, 1, dtype=int)

    fig, ax = pyplot.subplots()

    ax.plot(epochs_axis, losses['train'], 'ro--', label='Train loss')
    ax.plot(epochs_axis, losses['validation'], 'bo--', label='Validation loss')
    ax.plot(epochs_axis, losses['test'], 'go--', label='Testing loss')

    ax.set(xlabel='epochs', ylabel='J/n',
            title='Learning Curve')

    ax.legend()

    fig.savefig(file)
    pyplot.close(fig)
    

def plot_decision_boundary(file: str, data: np.ndarray, labels: np.ndarray, n: Network) -> None:
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

    fig = pyplot.figure()
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
    r'''Use the training set to normalize the combined data.

    Args:
        train: column vector (feat x samples) of data for mean, std computation
        combined: the data to normalize with std. normalization
    '''

    # mean and std of each feature
    # each row is one feature
    mean = np.mean(train, axis=1)
    std = np.std(train, axis=1, ddof=1)

    # turn into column vectors from 1d
    mean = mean[..., np.newaxis]
    std = std[..., np.newaxis]

    return (combined - mean)/std

def train(
    trainX: np.ndarray,
    trainY: np.ndarray,
    validationX: np.ndarray,
    validationY: np.ndarray,
    testX: np.ndarray,
    testY: np.ndarray,
    eta: float, 
    n: Network,
    ) -> dict[str, list[float, ...]]:
    r'''Train the network with the given learning parameter.

    It uses the arguments to construct and save plots for the report.

    Args:
        trainX, trainY: the set to train on + labels
        validationX, validationY: the set to determine when to stop training
        testX, testY: the set to evaluate model accuracy with
        eta: a learning parameter
        n: the neural network
    '''

    n_samples = trainX.shape[1]
    epochs = 0

    # add epoch 0 to this list
    losses = defaultdict(list)

    prev_err = n.error(n(validationX).T, validationY)/n_samples
    epsilon = 0.0002

    losses['train'].append(n.error(n(trainX).T, trainY)/n_samples)
    losses['validation'].append(prev_err)
    losses['test'].append(n.error(n(testX).T, testY)/n_samples)

    done = False
    while not done:
        # randomly select some order
        order = list(range(n_samples))
        np.random.shuffle(order)

        
        for j in order:
            x = trainX[:, j]
            x = x[..., np.newaxis]
            t = trainY[j]
            n.backward_online(x, t, eta)  # single sample training

        epochs += 1

        # stopping condition
        # error not decreasing much anymore from one epoch to the next
        # may continue if loss increases slightly since it's an absolute difference
        # but in validation the loss will usually continue to decrease after this
        fwd = n(validationX).transpose()
        err = n.error(fwd, validationY)/n_samples
        if np.abs(err - prev_err) <= epsilon:
            done = True  # just to print accuracy one final time
        else:
            prev_err = err

        
        # save for learning curve
        predict_train = n(trainX).transpose()
        predict_validation = fwd
        predict_test = n(testX).transpose()


        # print accuracy every 10 epochs
        if not epochs % 10 or done:
            train_accuracy = predict_train.round() == trainY
            train_accuracy = np.mean(train_accuracy)
            validation_accuracy = predict_validation.round() == validationY
            validation_accuracy = np.mean(validation_accuracy)
            test_accuracy = predict_test.round() == testY
            test_accuracy = np.mean(test_accuracy)

            prompt = 'Training set accuracy after {} epochs:'.format(epochs).ljust(45)
            print(prompt + '{:.4%}'.format(train_accuracy))

            prompt = 'Validation set accuracy after {} epochs:'.format(epochs).ljust(45)
            print(prompt + '{:.4%}'.format(validation_accuracy))

            prompt = 'Testing set accuracy after {} epochs:'.format(epochs).ljust(45)
            print(prompt + '{:.4%}'.format(test_accuracy))

            print()

        # save the loss/n_samples for later plot creation
        losses['train'].append(n.error(predict_train, trainY)/n_samples)
        losses['validation'].append(err)
        losses['test'].append(n.error(predict_test, testY)/n_samples)

        # kill condition
        # learning rate is too high and we're oscillating
        if epochs >= 200:
            done = True

    return losses


if __name__ == '__main__':
    main()
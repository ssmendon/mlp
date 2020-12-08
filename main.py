from mendon.nn import Perceptron, Network
import numpy as np
import numpy.matlib

def main():
    np.random.seed(0)

    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    X = X.transpose()

    t = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    n = Network(2, 2, 1)
    print(n(X))

    train1(X, t, n, 1, 20)

def train1(X, yhat, n, eta, n_epoch):
    n_samples = X.shape[-1]
    n_input = X.shape[0]
    n_output = 1
    
    # keep track of performance during training
    costs = np.zeros(shape=(n_epoch,1))

    print(n(X))

    for epoch in range(n_epoch):
        for i in range(n_samples):
            x0 = X[:,i]; yh = yhat[i]
            y = n(x0)  # prediction for one sample

            n.backward_online(x0, yh, eta)  # take step
        
        # ### Some niceness to see our progress
        # Calculate total cost after epoch
        predictions = n(X)  # predictions for entire set
        print(predictions)
        print(yhat)
        costs[epoch] = np.mean(n.error(predictions, yhat))  # mean cost per sample
        # report progress
        if ((epoch % 10) == 0) or (epoch == (n_epoch - 1)):
            #print(predictions.round())
            accuracy = np.mean(predictions.round() == yhat)  # current accuracy on entire set
            print('Training accuracy after epoch {}: {:.4%}'.format(epoch, accuracy))


if __name__ == '__main__':
    main()
from mendon.nn import Perceptron, Network
import numpy as np

def main():
    n = Network(2, 1, 2)
    
    x = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])

    # add bias
    bias = np.ones(
        (x.shape[0], 1)
    )

    x = np.concatenate((bias, x), axis=1)
    x = x.transpose()
    print(x)

    weights_j = np.array([
        [0.5, 1, 1],
        [-1.5, 1, 1]
    ])

    weights_k = np.array([
        [-1, 0.7, -0.4],
    ])

    weights = np.concatenate((weights_j, weights_k))
    weights = weights.transpose()

    net_j = np.matmul(weights[:,0:2].T, x)
    print()
    print(net_j)
    print()
    print(net_j[0,:])
    print()

    y = np.sign(net_j)
    y = np.concatenate((bias, y.T), axis=1).T  # reattach bias
    print('y =\n', y)
    print()

    net_k = np.matmul(weights[:,2:].T, y)
    outputs = np.sign(net_k)

    print(outputs)

    return


if __name__ == '__main__':
    main()
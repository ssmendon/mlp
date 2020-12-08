from mendon.nn import Perceptron, Network
import numpy as np
import numpy.matlib

def main():
    backprop_test()
    return

    x = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])

    t = np.array([
        -1, 1, 1, -1
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
    print()

    return

def backprop_test():
    np.random.seed(0)

    x = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ])

    t = np.array([
        -1, 1, 1, -1
    ])

    activation = lambda x: 1 / (1 - np.exp(-x))
    activation_deriv = lambda x: activation(x) * (1 - activation(x))

    error_func = lambda z, t: np.sum(np.square(t - z), axis=0)/2
    error_deriv = lambda y, z, t: -np.sum(t - z) * activation_deriv(y)

    
    errors = np.ones((1, len(x)))
    
    while np.linalg.norm(errors) >= 0.00001:
        # add bias
        bias = np.ones(
            (x.shape[0], 1)
        )

        x = np.concatenate((bias, x), axis=1)
        x = x.transpose()

        weights = np.matlib.rand(3, 3)

        net_j = np.matmul(weights[:,0:2].T, x)

        y = activation(net_j)

        y = np.concatenate((bias, y.T), axis=1).T  # reattach bias


        net_k = np.matmul(weights[:,2:].T, y)
        outputs = activation(net_k)

        # backprop starts here
        errors = error_func(outputs, t)

        break



if __name__ == '__main__':
    main()
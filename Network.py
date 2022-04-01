import numpy as np

#sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def linear(z):
    return z

def relu(z):
    if z < 0:
        return 0
    else:
        return z

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layers = sizes
        # biases = list of numpy arrays, one array for each hidden and output layer.
        # [array([layer 1]),
        # array([layer 2]), 
        # ...,
        # array([layer n])]
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]
        # list of lists of numpy arrays, each array gives weights for inputs to each node.
        # [[array([node 1 layer 1]),
        #   array([node 2 layer 1])]
        #  [array([node 1 layer 2]),
        #   array([node 2 layer 2]),
        #   array([node 3 layer 2])]
        #  ...,
        #  [array([node 1 layer j]),
        #   ...,
        #   array([node k layer j])]]
        self.weights = [np.random.randn(j, i) for i, j in zip(sizes[:-1], sizes[1:])]

    # activation is a numpy array of inputs to the first layer
    # returns a list of numpy arrays of the outputs from each layer
    # the final layer output is a np array found at a[-1].
    def forward(self, activation, func=relu):
        a = [activation]
        for b, w in zip(self.biases, self.weights):
            a.append(func(np.dot(w, a[-1]) + b))
        return a

import numpy as np

class Network():
    num_layers = 0
    layers = []
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layers = sizes
        # biases = list of numpy arrays, one array for each hidden and output layer.
        # [array([layer 1]),
        # array([layer 2]), 
        # ...,
        # array([layer n])]
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]
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

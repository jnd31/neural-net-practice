import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data
# nnfs for generating data, this is just so I remember what to import lmao

#sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def linear(z):
    return z

def relu(z):
    return np.maximum(0, z)

class Network():
    # sizes is a list containing the number of nodes in each layer, starting with input layer and ending with output layer.
    # a fn with 2 inputs, 4 neurons, and 3 outputs is thus [2, 4, 3]
    def __init__(self, sizes, zeros=False):
        self.num_layers = len(sizes)
        self.layers = sizes
        # biases = list of numpy arrays, one array for each hidden and output layer.
        # [array([layer 1]),
        # array([layer 2]), 
        # ...,
        # array([layer n])]
        if zeros:
            self.biases = [np.zeros(j, 1) for j in sizes[1:]]
            self.weights = [np.zeros(j, i) for i, j in zip(sizes[:-1], sizes[1:])]
        else:
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
    # should be [[node1, node1, node1], [node2, node2, node2]]
    # returns a list of numpy arrays of the outputs from each layer
    # the final layer output is a np array found at a[-1].
    def forward(self, activation, func=relu):
        a = [activation]
        for b, w in zip(self.biases, self.weights):
            a.append(func(np.dot(w, a[-1]) + b))
        return a

    # normalizes inputs with the exponential fn
    # inputs = numpy array of inputs to nn or outputs
    # returns a np array of np array of normalized probabilities
    def softmax(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    # |  ||
    # || |_
    def loss(self, outputs):
        pass
# FOR FUTURE REFERENCE, THIS IS HOW YOU MAKE AND SAVE DATA @ME
# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')
# fig.savefig('scatter plot.jpg')
# plt.show()
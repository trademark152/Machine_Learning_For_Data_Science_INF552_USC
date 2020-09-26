import numpy as np
from gate import MultiplyGate, AddGate
from output import Softmax, LSE
from layer import Tanh, Sigmoid

"""
initialize the parameters in the __init__ function.
"""
class Model:
    def __init__(self, layers_dim, activation_func, output_func):
        self.b = []
        self.W = []
        self.activation_func = activation_func
        self.output_func = output_func
        for i in range(len(layers_dim)-1):
            # initialize weights and biases
            # self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
            # self.b.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))
            self.W.append(self.randomizer(layers_dim[i], layers_dim[i + 1]))
            self.b.append(self.randomizer(1, layers_dim[i + 1]))


    # weight matrix via randomized initiation a*b dimension
    # a and b are the number of neurons in layer i and i+1 respectively
    # value specified between -0.01 and 0.01 as guided by the prompt
    def randomizer(self, a, b):
        return 0.02 * np.random.rand(a, b) - 0.01

    """
     First let’s implement the loss function we defined above. 
     It is just a forward propagation computation of out neural network.
     We use this to evaluate how well our model is doing:
     """

    def calculate_loss(self, X, y):
        m_Gate = MultiplyGate()
        a_Gate = AddGate()

        if self.activation_func == 'sigmoid':
            layer = Sigmoid()
        elif self.activation_func == 'tanh':
            layer = Tanh()

        if self.output_func == 'softmax':
            output = Softmax()
        elif self.output_func == 'lse':
            output = LSE()

        input = X
        # loop through each layer
        for i in range(len(self.W)):
            # X*W
            mul = m_Gate.forward(self.W[i], input)

            # X*W + b
            add = a_Gate.forward(mul, self.b[i])

            # nonlinear activation
            input = layer.forward(add)

        return output.eval_error(input, y)

    """
    implements batch gradient descent using the backpropagation
    """
    def train(self, X, y, num_passes=1000, lr=0.01, regularization=0.01, to_print=True):
        # add gates
        m_Gate = MultiplyGate()
        a_Gate = AddGate()

        # activate nonlinear layer
        if self.activation_func == 'sigmoid':
            layer = Sigmoid()
        elif self.activation_func == 'tanh':
            layer = Tanh()

        # activate output layer
        if self.output_func == 'softmax':
            output = Softmax()
        elif self.output_func == 'lse':
            output = LSE()

        # for each epoch
        for epoch in range(num_passes):
            # Forward propagation
            input = X
            forward = [(None, None, input)]

            # for each layer except the last one
            for i in range(len(self.W)):
                mul = m_Gate.forward(self.W[i], input)
                add = a_Gate.forward(mul, self.b[i])
                input = layer.forward(add)
                forward.append((mul, add, input))

            # last output of forward propagation is an array: num_samples * num_neurons_last_layer

            # Back propagation
            # derivative of cumulative error from output layer
            dfunc = output.calc_diff(forward[len(forward)-1][2], y)
            for i in range(len(forward)-1, 0, -1):
                # 1 layer consists of mul, add and layer
                dadd = layer.backward(forward[i][1], dfunc)

                # dLdb and dLdmul are functions of dLdadd
                db, dmul = a_Gate.backward(forward[i][0], self.b[i-1], dadd)
                dW, dfunc = m_Gate.backward(self.W[i-1], forward[i-1][2], dmul)

                # Add regularization terms (b1 and b2 don't have regularization terms)
                dW += regularization * self.W[i-1]

                # Gradient descent parameter update
                self.b[i-1] += -lr * db
                self.W[i-1] += -lr * dW

            if to_print and epoch % 100 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))



    """
    We also implement a helper function to calculate the output of the network.
    It does forward propagation as defined above and returns the class 
    with the highest probability.
    """
    def predict(self, X):
        m_Gate = MultiplyGate()
        a_Gate = AddGate()

        if self.activation_func == 'sigmoid':
            layer = Sigmoid()
        elif self.activation_func == 'tanh':
            layer = Tanh()

        if self.output_func == 'softmax':
            output = Softmax()
        elif self.output_func == 'lse':
            output = LSE()

        input = X
        for i in range(len(self.W)):
            mul = m_Gate.forward(self.W[i], input)
            add = a_Gate.forward(mul, self.b[i])
            input = layer.forward(add)

        if self.output_func == 'softmax':
            probs = output.eval(input)
            return np.argmax(probs, axis=1)
        elif self.output_func == 'lse':
            return (np.greater(input, 0.5))*1

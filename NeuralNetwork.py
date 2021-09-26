from Matrix import Matrix
from math import exp

"""Functions"""


def sigmoid(x):
    for i in range(x.rows):
        for j in range(x.cols):
            x.data[i][j] = 1 / (1 + exp(-x.data[i][j]))
    return x


def dsigmoid(x):
    for i in range(x.rows):
        for j in range(x.cols):
            x.data[i][j] = x.data[i][j] * (1 - x.data[i][j])
    return x


class NeuralNetwork:
    """Class to work with neural network"""

    def __init__(self, i_nodes, h_nodes, o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        self.learning_rate = 0.1

        """Create bias"""
        self.bias_ih = Matrix(self.h_nodes, 1)
        self.bias_ih.randomize()
        self.bias_ho = Matrix(self.o_nodes, 1)
        self.bias_ho.randomize()

        """Create weigths"""
        self.weigth_ih = Matrix(self.h_nodes, self.i_nodes)
        self.weigth_ih.randomize()
        self.weigth_ho = Matrix(self.o_nodes, self.h_nodes)
        self.weigth_ho.randomize()

    """Function that trains the neural network"""

    def train(self, arr, target):
        """FEEDFORWARD"""

        """IN ---> HIDDEN"""
        input = Matrix.arrayToMatrix(arr)
        hidden = Matrix.multiply(self.weigth_ih, input)
        hidden = Matrix.add(hidden, self.bias_ih)
        hidden = sigmoid(hidden)

        """HIDDEN ---> OUT"""
        output = Matrix.multiply(self.weigth_ho, hidden)
        output = Matrix.add(output, self.bias_ho)
        output = sigmoid(output)

        """BACKPROPAGATION"""

        """OUT ---> HIDDEN"""
        expected = Matrix.arrayToMatrix(target)
        output_error = Matrix.subtract(expected, output)
        d_output = dsigmoid(output)
        hidden_T = Matrix.transpose(hidden)

        gradient = Matrix.hadamard(d_output, output_error)
        gradient = Matrix.scalar_multiply(gradient, self.learning_rate)

        """Adjust Bias OUT ---> HIDDEN"""
        self.bias_ho = Matrix.add(self.bias_ho, gradient)

        """Adjust Weigths OUT ---> HIDDEN"""
        weigth_ho_deltas = Matrix.multiply(gradient, hidden_T)
        self.weigth_ho = Matrix.add(self.weigth_ho, weigth_ho_deltas)

        """HIDDEN ---> IN"""
        weigth_ho_T = Matrix.transpose(self.weigth_ho)
        hidden_error = Matrix.multiply(weigth_ho_T, output_error)
        d_hidden = dsigmoid(hidden)
        input_T = Matrix.transpose(input)

        gradient_H = Matrix.hadamard(d_hidden, hidden_error)
        gradient_H = Matrix.scalar_multiply(gradient_H, self.learning_rate)

        """Adjust Bias HIDDEN ---> IN"""
        self.bias_ih = Matrix.add(self.bias_ih, gradient_H)

        """Adjust Weigths HIDDEN ---> IN"""
        weigth_ih_deltas = Matrix.multiply(gradient_H, input_T)
        self.weigth_ih = Matrix.add(self.weigth_ih, weigth_ih_deltas)

    def predict(self, arr):
        """FEEDFORWARD"""

        """IN ---> HIDDEN"""
        input = Matrix.arrayToMatrix(arr)
        hidden = Matrix.multiply(self.weigth_ih, input)
        hidden = Matrix.add(hidden, self.bias_ih)
        hidden = sigmoid(hidden)

        """HIDDEN ---> OUT"""
        output = Matrix.multiply(self.weigth_ho, hidden)
        output = Matrix.add(output, self.bias_ho)
        output = sigmoid(output)
        output = Matrix.matrixToArray(output)

        return output

from NeuralNetwork import NeuralNetwork
import random

train = True
debug = True

nn = NeuralNetwork(2, 3, 1)

input = [[1, 1], [1, 0], [0, 1], [0, 0]]
output = [[0], [1], [1], [0]]

"""Neural Network Training"""
while train:
    for i in range(10000):
        index = random.randint(0, 3)
        nn.train(input[index], output[index])
        if debug:
            print(
                "\033[31m IN {} -----> OUT {} >>>> EXPECTED {} \033[0;0m".format(
                    input[index], nn.predict(input[index]), output[index]
                )
            )
    if nn.predict([0, 0])[0] < 0.04:
        if nn.predict([1, 0])[0] > 0.98:
            if debug:
                train = False
                print(
                    "\033[32m IN {} -----> OUT {} >>>> EXPECTED {} \033[0;0m".format(
                        input[index], nn.predict(input[index]), output[index]
                    )
                )

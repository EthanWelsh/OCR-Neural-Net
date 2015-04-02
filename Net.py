from Layer import *

class Net:

    def __init__(self, topology):

        if len(topology) < 3:
            print("Error. Invalid net topology. Net must have at least one input, output, and hidden layer.")
            exit(-1)

        self.layers = list()

        self.layers.append(Layer(topology[0]))

        self.input = self.layers[0]
        prevLayer = self.input

        for i in range(1, len(topology)):

            sizeOfLayer = topology[i]
            self.layers.append(Layer(sizeOfLayer, prevLayer))
            prevLayer = self.layers[len(self.layers) - 1]

        self.output = self.layers[len(topology) - 1]


    # Given inputs, propagate through net and produce outputs
    def feedForward(self, inputValues):
        return 0

    # Express target values for the current input and adjusts network accordingly
    def backProp(self, targetValues):
        return 0

    def getResults(self):
        return 0

    def __str__(self):
        s = ""
        for layer in self.layers:
            s += str(layer) + "\n"
        return s

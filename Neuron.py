import math
import random

class Neuron:

    # Normal constructor
    def __init__(self, neuronNumber, previousLayer = None):

        self.weights = list()

        if previousLayer == None:
            self.weights.append(1)
        else:
            self.neuronNumber = neuronNumber


            # Give a random weight between 0 and 1 for each input
            for i in range(0, len(previousLayer)):
                self.weights.append(random.uniform(0.0, 1.0))

    def getWeights(self):
        return self.weights

    def setWeights(self, newWeights):

        if len(newWeights) != len(self.weights):
            print("You gave the incorrect number of weights. This neuron only has", len(self.weights), "and you gave", len(newWeights), "weights.")

        for i in range (0, len(newWeights)):
            self.weights[i] = newWeights[i]

    def getOutputValue(self):
        return self.outputValue

    def setOutputValue(self, outputValue):
        self.outputValue = outputValue

    def calculateOutput(self, previousLayer):

        sum = 0
        neuronNumber = 0

        for neuron in previousLayer:
            sum += (neuron * self.connections[neuronNumber].weight)
            neuronNumber += 1

        return sigmoid(sum)

    def sigmoid (self, x):
        return math.tanh(x)

    def dsigmoid (self, y):
        return 1 - y**2

    def __str__(self):

        s = "( "

        for w in range(0, len(self.weights)):
            s += "{0:.2f}".format(self.weights[w]) + " "

        s += ")"

        return s
import math
import random

class Neuron:

    eta = .15 # learning rate
    alpha = .5 #momentum

    # Normal constructor
    def __init__(self, neuronNumber, previousLayer = 0):

        self.weights = list()
        self.deltaWeights = list()

        self.gradient = 0.0

        self.outputValue = 0

        if previousLayer == 0:
            self.weights.append(1)
        else:
            self.neuronNumber = neuronNumber


            # Give a random weight between 0 and 1 for each input
            for i in range(0, previousLayer):
                self.weights.append(random.uniform(0.0, 1.0))

        for w in self.weights:
            self.deltaWeights.append(0)

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

    def feedForward(self, previousLayer):

        sum = 0.0

        neuronNumber = 0
        for neuron in previousLayer:
            sum += neuron.getOutputValue() * self.weights[neuronNumber] #output value * connection weight
            neuronNumber += 1

        return self.sigmoid(sum) # apply transfer function

    def sigmoid (self, x):
        return math.tanh(x)

    def dsigmoid (self, y):
        return 1 - y**2

    def calculateOutputGradients(self, targetValue):

        delta = targetValue - self.getOutputValue()
        self.gradient = delta * self.dsigmoid(self.getOutputValue())

    def calculateHiddenGradients(self, nextLayer):

        sumDerivativeOfWeights = self.sumDOW(nextLayer)
        self.gradient = sumDerivativeOfWeights * self.dsigmoid(self.getOutputValue())

    def sumDOW(self, nextLayer):

        sum = 0.0

        for neuron in range(0, len(nextLayer) - 1):
            sum += self.weights[neuron] * nextLayer[neuron].gradient

        return sum

    def updateInputWeights(self, previousLayer):
        for n in range(0, len(previousLayer)):
            current = previousLayer[n]
            oldDeltaWeight = current.weights[self.neuronNumber] # weight from current to us
            newDeltaWeight = self.eta * current.getOutputValue() * self.gradient + self.alpha * oldDeltaWeight

            current.deltaWeights[self.neuronNumber] = newDeltaWeight
            current.weights[self.neuronNumber] += newDeltaWeight

    def __str__(self):

        s = "( "

        for w in range(0, len(self.weights)):
            s += "{0:.2f}".format(self.weights[w]) + " "

        s += ")"

        return s
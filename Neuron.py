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

    # Get the output value
    def getOutputValue(self):
        return self.outputValue

    # Manually set the output value
    def setOutputValue(self, outputValue):
        self.outputValue = outputValue

    # Given inputs, will apply connection weights to these input and return the resulting sigmoidial value
    def feedForward(self, previousLayer):

        sum = 0.0

        neuronNumber = 0
        for neuron in previousLayer:
            sum += neuron.getOutputValue() * self.weights[neuronNumber] #output value * connection weight
            neuronNumber += 1

        return self.sigmoid(sum) # apply transfer function

    # Transfer function
    def sigmoid (self, x):
        return math.tanh(x)

    # Transfer function derivative
    def dsigmoid (self, y):
        return 1 - y**2

    # Calculates the gradients of the output layer for use in back propogation
    def calculateOutputGradients(self, targetValue):

        delta = targetValue - self.getOutputValue()
        self.gradient = delta * self.dsigmoid(self.getOutputValue())

    # Calculates the gradients from the hidden layers for use in back propogation
    def calculateHiddenGradients(self, nextLayer):

        sumDerivativeOfWeights = self.sumDOW(nextLayer)
        self.gradient = sumDerivativeOfWeights * self.dsigmoid(self.getOutputValue())

    # Finds the sum of the derivate of weights for use in back propogation
    def sumDOW(self, nextLayer):

        sum = 0.0
        for neuron in range(0, len(nextLayer) - 1):
            sum += self.weights[neuron] * nextLayer[neuron].gradient
        return sum

    # Given the previous layer, will adjust input weights in accordance with gradient and delta weight
    def updateInputWeights(self, previousLayer):
        for n in range(0, len(previousLayer)):
            current = previousLayer[n]

            print(self.neuronNumber)

            try: ### TODO TODO TODO
                oldDeltaWeight = current.weights[self.neuronNumber] # weight from current to us
            except IndexError:
                print("INDEX ERROR ON", self.neuronNumber)

            print(oldDeltaWeight)

            newDeltaWeight = self.eta * current.getOutputValue() * self.gradient + self.alpha * oldDeltaWeight

            current.deltaWeights[self.neuronNumber] = newDeltaWeight
            current.weights[self.neuronNumber] += newDeltaWeight

    # Get string representation of neuron
    def __str__(self):

        s = "( "

        for w in range(0, len(self.weights)):
            s += "{0:.2f}".format(self.weights[w]) + " "

        s += ")"

        return s
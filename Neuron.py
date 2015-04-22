import random
import math


class Connection:

    def __init__(self, w=0.0, d=0.0):
        self.weight = w
        self.deltaWeight = d


class Neuron:

    eta = 0.15      # overall net learning rate
    alpha = 0.5     # momentum, multiplier of last deltaWeight

    def __init__(self, numOutputs, myIndex):
        self.m_outputWeights = []

        for c in range(0, numOutputs):
            self.m_outputWeights.append(Connection())
            self.m_outputWeights[len(self.m_outputWeights) - 1].weight = random.uniform(-1.0, 1.0)

        self.m_myIndex = myIndex


    def getOutputVal(self):
        return self.m_outputVal


    def setOutputVal(self, val):
        self.m_outputVal = val


    def updateInputWeights(self, prevLayer):
        # The weights to be updated are in the Connection container in the neurons in the preceding layer
        for n in range(0, len(prevLayer)):
            neuron = prevLayer[n]
            oldDeltaWeight = neuron.m_outputWeights[self.m_myIndex].deltaWeight

            newDeltaWeight = self.eta * neuron.getOutputVal() * self.m_gradient + self.alpha * oldDeltaWeight

            neuron.m_outputWeights[self.m_myIndex].deltaWeight = newDeltaWeight
            neuron.m_outputWeights[self.m_myIndex].weight += newDeltaWeight


    def sumDOW(self, nextLayer):
        sum = 0.0

        # Sum our contributions of the errors at the nodes we feed.
        for n in range(0, len(nextLayer) - 1):
            sum += self.m_outputWeights[n].weight * nextLayer[n].m_gradient
        return sum


    def transferFunction(self, x):
        return math.tanh(x)


    def transferFunctionDerivative(self, x):
        return 1.0 - x * x


    def calcHiddenGradients(self, nextLayer):
        dow = self.sumDOW(nextLayer)
        self.m_gradient = dow * self.transferFunctionDerivative(self.m_outputVal)


    def calcOutputGradients(self, targetVal):
        self.delta = targetVal - self.m_outputVal
        self.m_gradient = self.delta * self.transferFunctionDerivative(self.m_outputVal)


    def feedForward(self, prevLayer):
        sum = 0.0
        # Sum the previous layer's outputs (which are our inputs) include the bias node from the previous layer.

        for n in range(0, len(prevLayer)):
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[self.m_myIndex].weight

        self.m_outputVal = self.transferFunction(sum)


    def __str__(self):
        ret = "( "
        for w in self.m_outputWeights:
            ret += "{0:.2f}".format(w.weight) + " "
        ret += ")"
        return ret
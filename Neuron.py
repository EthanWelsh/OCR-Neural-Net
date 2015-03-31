import math

class Neuron:

    def __init__(self, neuronNumber):

        self.neuronNumber = 0
        self.outputValue = 0
        self.weights = []

    def getOutputValue(self):
        return self.outputValue

    def setOutputValue(self, output):
        self.outputValue = output

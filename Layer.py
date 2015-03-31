from Neuron import *

class Layer:

    def __init__(self, numberOfNeurons):
        self.neurons = []

        for x in range(0, numberOfNeurons):
            self.neurons = Neuron()


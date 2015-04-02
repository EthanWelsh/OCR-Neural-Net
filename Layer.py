from Neuron import *

class Layer:

    # Normal constructor
    def __init__(self, numberOfNeurons, previousLayer = None):

        self.neurons = list()

        if previousLayer == None:


            for x in range(0, numberOfNeurons + 1): # + 1 for bias neuron
                self.neurons.append(Neuron(x))
        else:
            for x in range(0, numberOfNeurons + 1): # + 1 for bias neuron
                self.neurons.append(Neuron(x, len(previousLayer)))


        self.biasNeuron = self.neurons[numberOfNeurons]
        self.biasNeuron.setOutputValue(1.0)


    def getOutputs(self):

        outputs = list()

        for neuron in (0, len(self.neurons) - 1):
            outputs.append(self.neurons[neuron].getOutputValue())

        return outputs

    def __str__(self):

        s = ""
        for n in self.neurons:
            s += str(n) + " "

        return s

    def __getitem__(self, item):
        return self.neurons[item]

    def __len__(self):
        return len(self.neurons)
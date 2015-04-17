from Layer import *


class Net:
    m_error = 0.0

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

        for n in range(0, len(self.input) - 1):  # set inputs
            self.input[n].setOutputValue(inputValues[n])

        for layer in range(1, len(self.layers)):  # for every layer
            for neuron in self.layers[layer]:  # feedForward every neuron
                neuron.feedForward(self.layers[layer - 1])

    # Express target values for the current input and adjusts network accordingly
    def backProp(self, targetValues):

        # Calculate RMS of output neuron errors
        for outputNeuron in range(0, len(self.output) - 1):
            delta = targetValues[outputNeuron] - self.output[outputNeuron].getOutputValue()
            Net.m_error += delta * delta

        Net.m_error /= len(self.output) - 1
        Net.m_error = math.sqrt(Net.m_error)

        # Calculate output layer gradients
        for outputNeuron in range(0, len(self.output) - 1):
            self.output[outputNeuron].calculateOutputGradients(targetValues[outputNeuron])

        # Calculate hidden layer gradients
        layerNumber = len(self.layers) - 2

        while layerNumber > 0:
            thisLayer = self.layers[layerNumber]
            nextLayer = self.layers[layerNumber + 1]

            for neuron in thisLayer:
                neuron.calculateHiddenGradients(nextLayer)

            layerNumber -= 1

        # Update connection weights
        layerNumber = len(self.layers) - 1
        while layerNumber > 0:
            currentLayer = self.layers[layerNumber]
            previousLayer = self.layers[layerNumber - 1]

            for neuron in currentLayer:
                neuron.updateInputWeights(previousLayer)

            layerNumber -= 1

    def getResults(self):
        return self.output

    def __str__(self):
        s = ""
        for layer in self.layers:
            s += str(layer) + "\n"
        return s

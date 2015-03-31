from Layer import *

class Net:

    def __init__(self, layerDescriptors):

        self.layers = []

        for sizeOfLayer in layerDescriptors:
            self.layers.append(Layer(sizeOfLayer))

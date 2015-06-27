import random
import math


class Connection:
    def __init__(self, w=0.0, d=0.0):
        self.weight = w
        self.deltaWeight = d


class Neuron:
    eta = 0.15  # overall net learning rate
    alpha = 0.5  # momentum, multiplier of last deltaWeight

    def __init__(self, num_outputs, my_index):
        self.output_weights = []
        self.output_val = None
        self.gradient = None
        self.delta = None

        for c in range(0, num_outputs):
            self.output_weights.append(Connection())
            self.output_weights[len(self.output_weights) - 1].weight = random.uniform(-1.0, 1.0)

        self.my_index = my_index

    def get_output_val(self):
        return self.output_val

    def set_output_val(self, val):
        self.output_val = val

    def update_input_weights(self, prev_layer):
        # The weights to be updated are in the Connection container in the neurons in the preceding layer
        for n in range(0, len(prev_layer)):
            neuron = prev_layer[n]
            old_delta_weight = neuron.output_weights[self.my_index].deltaWeight

            new_delta_weight = self.eta * neuron.get_output_val() * self.gradient + self.alpha * old_delta_weight

            neuron.output_weights[self.my_index].deltaWeight = new_delta_weight
            neuron.output_weights[self.my_index].weight += new_delta_weight

    def sum_dow(self, next_layer):
        """Sum our contributions of the errors at the nodes we feed"""
        error_sum = 0.0
        for n in range(0, len(next_layer) - 1):
            error_sum += self.output_weights[n].weight * next_layer[n].gradient
        return error_sum

    def calc_hidden_gradients(self, nextLayer):
        dow = self.sum_dow(nextLayer)
        self.gradient = dow * self.transfer_function_derivative(self.output_val)

    def calc_output_gradients(self, targetVal):
        self.delta = targetVal - self.output_val
        self.gradient = self.delta * self.transfer_function_derivative(self.output_val)

    def feed_forward(self, prevLayer):
        """Sum the previous layer's outputs (which are our inputs) include the bias node from the previous layer"""
        output_sum = 0.0
        for n in range(0, len(prevLayer)):
            output_sum += prevLayer[n].get_output_val() * prevLayer[n].output_weights[self.my_index].weight
        self.output_val = self.transfer_function(output_sum)

    def __str__(self):
        ret = "( "
        for w in self.output_weights:
            ret += "{0:.2f}".format(w.weight) + " "
        ret += ")"
        return ret

    @staticmethod
    def transfer_function(x):
        return math.tanh(x)

    @staticmethod
    def transfer_function_derivative(x):
        return 1.0 - x * x

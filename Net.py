from Neuron import *

Layer = []


class Net:
    def __init__(self, topology):
        self.m_error = 0.0
        self.layers = []

        num_layers = len(topology)

        for layer_num in range(0, num_layers):
            self.layers.append([])

            if layer_num == len(topology) - 1:
                num_outputs = 0
            else:
                num_outputs = topology[layer_num + 1]

            # We have a new layer, now fill it with neurons, and add a bias neuron in each layer.
            for neuron_num in range(0, topology[layer_num] + 1):
                last_index = len(self.layers) - 1
                self.layers[last_index].append(Neuron(num_outputs, neuron_num))

            # Force the bias node's output to 1.0 (last neuron pushed in the latest added layer):
            last_index = len(self.layers) - 1
            last_neuron_index = len(self.layers[last_index]) - 1
            self.layers[last_index][last_neuron_index].set_output_val(1.0)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[len(self.layers) - 1]

    def get_results(self):
        results = []
        output_layer = self.output_layer

        for output_neuron in range(0, len(output_layer) - 1):
            results.append(output_layer[output_neuron].get_output_val())
        return results

    def feed_forward(self, input_vals):
        assert (len(input_vals) == len(self.input_layer) - 1)

        # Assign (latch) the input values into the input neurons
        for i in range(0, len(input_vals)):
            self.input_layer[i].set_output_val(input_vals[i])

        # forward propagate
        for layer_num in range(1, len(self.layers)):
            prevLayer = self.layers[layer_num - 1]
            for n in range(0, len(self.layers[layer_num]) - 1):
                self.layers[layer_num][n].feed_forward(prevLayer)
                n += 1
            layer_num += 1

    def back_prop(self, target_vals):
        # Calculate overall net error (RMS of output neuron errors)
        self.m_error = 0.0

        for n in range(0, len(self.output_layer) - 1):
            delta = target_vals[n] - self.output_layer[n].get_output_val()
            self.m_error += delta * delta

        self.m_error /= len(self.output_layer) - 1  # get average error squared
        self.m_error = math.sqrt(self.m_error)  # RMS

        # Calculate output layer gradients
        for n in range(0, len(self.output_layer) - 1):
            self.output_layer[n].calc_output_gradients(target_vals[n])

        # Calculate hidden layer gradients
        layer_num = len(self.layers) - 2

        while layer_num > 0:
            hidden_layer = self.layers[layer_num]
            next_layer = self.layers[layer_num + 1]

            for n in range(0, len(hidden_layer)):
                hidden_layer[n].calc_hidden_gradients(next_layer)

            layer_num -= 1

        layer_num = len(self.layers) - 1

        while layer_num > 0:
            layer = self.layers[layer_num]
            prev_layer = self.layers[layer_num - 1]

            for n in range(0, len(layer) - 1):
                layer[n].update_input_weights(prev_layer)
            layer_num -= 1

    def __str__(self):
        ret = ""
        for layer in self.layers:
            for neuron in layer:
                ret += str(neuron)
            ret += "\n"

        return ret

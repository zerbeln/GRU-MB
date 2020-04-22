from neural_net import NeuralNetwork
import numpy as np

def main():
    num_inputs = 8
    num_hnodes = 10
    num_outputs = 2
    time_steps = 1

    nn = NeuralNetwork(num_inputs, num_hnodes, num_outputs, time_steps)

    # Quick Test for GRU-MB architecture
    state_vec = np.ones(num_inputs)
    layer1_weights = np.ones((num_inputs+num_outputs+1)*num_hnodes)
    layer2_weights = np.ones((num_hnodes+1)*num_outputs)
    igate_weights = np.ones((num_inputs+1)*num_inputs)
    rgate_weights = np.ones((num_outputs+1)*num_outputs)
    wgate_weights = np.ones((num_outputs+1)*num_outputs)

    nn.run_neural_network(0, state_vec, layer1_weights, layer2_weights, igate_weights, rgate_weights, wgate_weights)


main()

import numpy as np


class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs, num_tsteps, mean=0, std=1):

        # GRU Properties
        self.n_inputs = int(num_inputs)
        self.n_outputs = int(num_outputs)
        self.bias = 1.0
        self.n_tsteps = num_tsteps

        # Memory Block
        self.memory_block = np.mat(np.zeros((num_tsteps, num_outputs)))

        # Feedforward Neural Network
        self.ffn_inputs = int(num_inputs + num_outputs)
        self.ffn_hnodes = int(num_hidden)  # Number of nodes in hidden layer
        self.n_layer1_weights = int((self.ffn_inputs + 1)*self.ffn_hnodes)
        self.n_layer2_weights = int((self.ffn_hnodes + 1)*self.n_outputs)
        self.layer1_weights = np.mat(np.zeros(self.n_layer1_weights))
        self.layer2_weights = np.mat(np.zeros(self.n_layer2_weights))
        self.in_layer = np.mat(np.zeros(self.ffn_inputs+1))
        self.hid_layer = np.mat(np.zeros(self.ffn_hnodes))
        self.out_layer = np.mat(np.zeros(self.n_outputs))

        # Input Gate
        self.num_igw = (num_inputs+1)*num_inputs
        self.igate_inputs = np.mat(np.zeros(num_inputs+1))
        self.igate_weights = np.mat(np.zeros(self.num_igw))
        self.igate_outputs = np.mat(np.zeros(num_inputs))

        # Read Gate
        self.num_rgw = (num_outputs+1)*num_outputs
        self.rgate_inputs = np.mat(np.zeros(num_outputs+1))
        self.rgate_weights = np.mat(np.zeros(self.num_rgw))
        self.rgate_outputs = np.mat(np.zeros(num_outputs))

        # Write Gate
        self.num_wgw = (num_outputs+1)*num_outputs
        self.wgate_inputs = np.mat(np.zeros(num_outputs+1))
        self.wgate_weights = np.mat(np.zeros(self.num_wgw))
        self.wgate_outputs = np.mat(np.zeros(num_outputs))

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """

        # Feedforward Network
        self.layer1_weights = np.mat(np.zeros(self.n_layer1_weights))
        self.layer2_weights = np.mat(np.zeros(self.n_layer2_weights))
        self.in_layer = np.mat(np.zeros(self.ffn_inputs+1))
        self.hid_layer = np.mat(np.zeros(self.ffn_hnodes))
        self.out_layer = np.mat(np.zeros(self.n_outputs))

        # Memory Block
        self.memory_block = np.mat(np.zeros((self.n_tsteps, self.n_outputs)))

        # Input Gate
        self.igate_inputs = np.mat(np.zeros(self.n_inputs + 1))
        self.igate_weights = np.mat(np.zeros(self.num_igw))
        self.igate_outputs = np.mat(np.zeros(self.n_inputs))

        # Read Gate
        self.rgate_inputs = np.mat(np.zeros(self.n_outputs + 1))
        self.rgate_weights = np.mat(np.zeros(self.num_rgw))
        self.rgate_outputs = np.mat(np.zeros(self.n_outputs))

        # Write Gate
        self.wgate_inputs = np.mat(np.zeros(self.n_outputs + 1))
        self.wgate_weights = np.mat(np.zeros(self.num_wgw))
        self.wgate_outputs = np.mat(np.zeros(self.n_outputs))

    def get_inputs(self, state_vec):  # Get inputs from state-vector
        """
        Assign inputs from rover sensors to the input layer of the NN
        :param state_vec: Inputs from rover sensors
        :param rov_id: Current rover
        :return: None
        """

        for i in range(self.n_inputs):
            self.in_layer[i] = state_vec[i]

    def get_weights(self, wl1, wl2, igw, rgw, wgw):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :return: None
        """

        # First layer of FFN weights
        for w in range(self.n_layer1_weights):
            self.layer1_weights[0, w] = wl1[w]
        self.layer1_weights = np.reshape(self.layer1_weights, [self.ffn_inputs+1, self.ffn_hnodes])

        # Second layer of FFN weights
        for w in range(self.n_layer2_weights):
            self.layer2_weights[0, w] = wl2[w]
        self.layer2_weights = np.reshape(self.layer2_weights, [self.ffn_hnodes+1, self.n_outputs])

        # Input gate weights
        for w in range(self.num_igw):
            self.igate_weights[0, w] = igw[w]
        self.igate_weights = np.reshape(self.igate_weights, [self.n_inputs+1, self.n_inputs])

        # Read gate weights
        for w in range(self.num_rgw):
            self.rgate_weights[0, w] = rgw[w]
        self.rgate_weights = np.reshape(self.rgate_weights, [self.n_outputs+1, self.n_outputs])

        # Write Gate Weights
        for w in range(self.num_wgw):
            self.wgate_weights[0, w] = wgw[w]
        self.wgate_weights = np.reshape(self.wgate_weights, [self.n_outputs+1, self.n_outputs])

    def run_input_gate(self, state_vec):
        """
        Process sensor inputs through input gate
        :param state_vec:
        :return:
        """
        for i in range(self.n_inputs):
            self.igate_inputs[0, i] = state_vec[i]
        self.igate_inputs[0, self.n_inputs] = self.bias

        self.igate_outputs = np.dot(self.igate_inputs, self.igate_weights)
        for i in range(self.n_inputs):
            self.igate_outputs[0, i] = self.sigmoid(self.igate_outputs[0, i])


    def run_read_gate(self):
        """
        Process read gate information
        :return:
        """

        for i in range(self.n_outputs):
            self.rgate_inputs[0, i] = self.out_layer[0, i]
        self.rgate_inputs[0, self.n_outputs] = self.bias

        self.rgate_outputs = np.dot(self.rgate_inputs, self.rgate_weights)
        for i in range(self.n_outputs):
            self.rgate_outputs[0, i] = self.sigmoid(self.rgate_outputs[0, i])

    def run_write_gate(self, tstep):
        """
        Process write gate
        :return:
        """

        for i in range(self.n_outputs):
            self.wgate_inputs[0, i] = self.out_layer[0, i]
        self.wgate_inputs[0, self.n_outputs] = self.bias

        self.wgate_outputs = np.dot(self.wgate_inputs, self.wgate_weights)
        for i in range(self.n_outputs):
            self.wgate_outputs[0, i] = self.sigmoid(self.wgate_outputs[0, i])
            self.memory_block[tstep, i] = self.wgate_outputs[0, i]

    def create_block_inputs(self):
        """
        Create the input layer for the block (feedforward network)
        :return:
        """

        for i in range(self.n_inputs):
            self.in_layer[0, i] = self.igate_outputs[0, i]

        for j in range(self.n_outputs):
            self.in_layer[0, self.n_inputs + j] = self.rgate_outputs[0, j]

        self.in_layer[0, self.n_inputs + self.n_outputs] = self.bias

    def feedforward_network(self, tstep):
        """
        Run NN to receive rover action outputs
        :return: None
        """

        self.create_block_inputs()
        self.hid_layer = np.dot(self.in_layer, self.layer1_weights)

        for i in range(self.ffn_hnodes):
            self.hid_layer[0, i] = self.sigmoid(self.hid_layer[0, i])

        self.hid_layer = np.insert(self.hid_layer, 0, self.bias, axis=1)
        self.out_layer = np.dot(self.hid_layer, self.layer2_weights)

        for i in range(self.n_outputs):
            self.out_layer[0, i] = self.sigmoid(self.out_layer[0, i])

        self.run_write_gate(tstep)

    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """

        tanh = (2/(1 + np.exp(-2*inp)))-1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: sigmoid value
        """

        sig = 1/(1 + np.exp(-inp))

        return sig

    def run_neural_network(self, tstep, state_vec, wl1, wl2, igw, rgw, wgw):
        """
        Run through NN for given rover
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :param rover_id: Rover identifier
        :return: None
        """
        self.get_weights(wl1, wl2, igw, rgw, wgw)
        self.run_input_gate(state_vec)
        self.run_read_gate()
        self.feedforward_network(tstep)

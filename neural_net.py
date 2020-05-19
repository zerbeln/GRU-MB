import numpy as np


class NeuralNetwork:

    def __init__(self, parameters):

        # GRU Properties
        self.n_inputs = int(parameters["n_inputs"])
        self.n_hnodes = int(parameters["n_hnodes"])  # Number of nodes in hidden layer
        self.n_outputs = int(parameters["n_outputs"])
        self.mem_block_size = int(parameters["mem_block_size"])
        self.bias = 1.0

        # Feedforward Neural Network Weights
        self.out_layer_weights = np.mat(np.zeros(self.mem_block_size))
        self.out_bias_weights = np.mat(np.zeros(self.n_outputs))

        # Feedforward Neural Network Layers
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))

        # Input Gate
        self.igate_inputs = np.mat(np.zeros(self.n_inputs))
        self.igate_weights = {}
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Block Input
        self.block_weights = {}
        self.block_input = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))

        # Read Gate
        self.rgate_inputs = np.mat(np.zeros(self.mem_block_size))
        self.rgate_weights = {}
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Write Gate
        self.wgate_inputs = np.mat(np.zeros(self.mem_block_size))
        self.wgate_weights = {}
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))

    def reset_nn(self):  # Clear current network
        """
        Clears neural network arrays so that they all contain zeros
        :return: None
        """

        # Feedforward Network Weights
        self.out_layer_weights = np.mat(np.zeros(self.mem_block_size))
        self.out_bias_weights = np.mat(np.zeros(self.n_outputs))

        # Feedforward Network Layers
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))

        # Input Gate
        self.igate_inputs = np.mat(np.zeros(self.n_inputs))
        self.igate_weights = {}
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Block Input
        self.block_input = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))
        self.block_weights = {}

        # Read Gate
        self.rgate_inputs = np.mat(np.zeros(self.mem_block_size))
        self.rgate_weights = {}
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Write Gate
        self.wgate_inputs = np.mat(np.zeros(self.mem_block_size))
        self.wgate_weights = {}
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))

    def get_weights(self, weights):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :return: None
        """

        # Output weights
        self.out_bias_weights = weights["b_out"]
        self.out_layer_weights = weights["p_out"]

        # Input gate weights
        n_mat = np.mat(weights["n_igate"])
        self.igate_weights["K"] = np.reshape(np.mat(weights["k_igate"]), [self.mem_block_size, 1])  # nx1
        self.igate_weights["R"] = np.reshape(np.mat(weights["r_igate"]), [self.mem_block_size, 1])  # nx1
        self.igate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.igate_weights["b"] = np.reshape(np.mat(weights["b_igate"]), [self.mem_block_size, 1])  # nx1

        # Block Input
        n_mat = np.mat(weights["n_block"])
        self.block_weights["K"] = np.reshape(np.mat(weights["k_block"]), [self.mem_block_size, 1])  # nx1
        self.block_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.block_weights["b"] = np.reshape(np.mat(weights["b_block"]), [self.mem_block_size, 1])  # nx1

        # Read gate weights
        n_mat = np.mat(weights["n_rgate"])
        self.rgate_weights["K"] = np.reshape(np.mat(weights["k_rgate"]), [self.mem_block_size, 1])  # nx1
        self.rgate_weights["R"] = np.reshape(np.mat(weights["r_rgate"]), [self.mem_block_size, 1])  # nx1
        self.rgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.rgate_weights["b"] = np.reshape(np.mat(weights["b_rgate"]), [self.mem_block_size, 1])  # nx1

        # Write Gate Weights
        n_mat = np.mat(weights["n_wgate"])
        self.wgate_weights["K"] = np.reshape(np.mat(weights["k_wgate"]), [self.mem_block_size, 1])
        self.wgate_weights["R"] = np.reshape(np.mat(weights["r_wgate"]), [self.mem_block_size, 1])
        self.wgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.wgate_weights["b"] = np.reshape(np.mat(weights["b_wgate"]), [self.mem_block_size, 1])  # nx1

    def run_input_gate(self, state_vec, mem_block):
        """
        Process sensor inputs through input gate
        :param mem:
        :param state_vec:
        :return:
        """

        x = np.mat(state_vec)  # 1x1
        y = np.mat(self.prev_out_layer)  # 1x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.igate_weights["b"]  # nx1

        X = np.dot(self.igate_weights["K"], x)  # nx1 * 1x1
        Y = np.dot(self.igate_weights["R"], y)  # nx1 * 1x1
        M = np.dot(self.igate_weights["N"], m)  # nxn * nx1

        self.igate_outputs = X + Y + M + b
        for i in range(self.mem_block_size):
            self.igate_outputs[i, 0] = self.sigmoid(self.igate_outputs[i, 0])

    def run_read_gate(self, state_vec, mem_block):
        """
        Process read gate information
        :return:
        """

        x = np.mat(state_vec)  # 1x1
        y = np.mat(self.prev_out_layer)  # 1x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.rgate_weights["b"]  # nx1

        X = np.dot(self.rgate_weights["K"], x)  # nx1 * 1x1
        Y = np.dot(self.rgate_weights["R"], y)  # nx1 * 1x1
        M = np.dot(self.rgate_weights["N"], m)  # nxn * 1x1

        self.rgate_outputs = X + Y + M + b
        for i in range(self.mem_block_size):
            self.rgate_outputs[i, 0] = self.sigmoid(self.rgate_outputs[i, 0])

    def run_write_gate(self, state_vec, mem_block):
        """
        Process write gate
        :return:
        """

        x = np.mat(state_vec)  # 1x1
        y = np.mat(self.prev_out_layer)  # 1x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.wgate_weights["b"]  # nx1

        X = np.dot(self.wgate_weights["K"], x)  # nx1 * 1x1
        Y = np.dot(self.wgate_weights["R"], y)  # nx1 * 1x1
        M = np.dot(self.wgate_weights["N"], m)  # nxn * nx1

        self.wgate_outputs = X + Y + M + b
        for i in range(self.mem_block_size):
            self.wgate_outputs[i, 0] = self.sigmoid(self.wgate_outputs[i, 0])

    def create_block_inputs(self, state_vec, mem_block):
        """
        Create the input layer for the block (feedforward network)
        :return:
        """

        x = np.mat(state_vec)  # 1x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.block_weights["b"]  # nx1

        X = np.dot(self.block_weights["K"], x)  # nx1 * 1x1
        M = np.dot(self.block_weights["N"], m)  # nxn * nx1

        self.block_input = X + M + b

    def feedforward_network(self, mem_block):
        """
        Run NN to receive rover action outputs
        :return: None
        """

        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])
        x1 = np.multiply(self.rgate_outputs, m)
        x2 = np.multiply(self.block_input, self.igate_outputs)

        self.block_output = x1 + x2

        self.out_layer = np.dot(self.out_layer_weights, self.block_output) + self.out_bias_weights

        for v in range(self.n_outputs):
            self.out_layer[0, v] = self.sigmoid(self.out_layer[0, v])

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

    def run_neural_network(self, state_vec, mem_block):
        """
        Run through NN for given rover
        :param mem_block:
        :param state_vec:
        :param rover_input: Inputs from rover sensors
        :param weight_vec:  Weights from CCEA
        :param rover_id: Rover identifier
        :return: None
        """
        self.run_input_gate(state_vec, mem_block)
        self.run_read_gate(state_vec, mem_block)
        self.create_block_inputs(state_vec, mem_block)
        self.feedforward_network(mem_block)
        self.run_write_gate(state_vec, mem_block)
        self.prev_out_layer = self.out_layer.copy()

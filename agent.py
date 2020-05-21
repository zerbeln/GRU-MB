import numpy as np

class Agent:

    def __init__(self, parameters):
        self.mem_block_size = parameters["mem_block_size"]
        self.mem_block = np.zeros(parameters["mem_block_size"])

    def reset_mem_block(self):
        self.mem_block = np.zeros(self.mem_block_size)

    def update_memory(self, nn_hblock, nn_wgate):
        hblock = np.reshape(nn_hblock, [1, self.mem_block_size])
        for v in range(self.mem_block_size):
            hblock[0, v] = self.tanh(hblock[0, v])
        wgate = np.reshape(nn_wgate, [1, self.mem_block_size])
        var = np.multiply(wgate, hblock)

        self.mem_block = self.mem_block + var

    def tanh(self, inp):  # Tanh function as activation function
        """
        NN activation function
        :param inp: Node value in NN (pre-activation function)
        :return: tanh value
        """

        tanh = (2/(1 + np.exp(-2*inp)))-1

        return tanh


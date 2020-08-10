import numpy as np
from parameters import parameters as p


class Agent:

    def __init__(self):
        self.mem_block_size = p["mem_block_size"]
        self.mem_block = np.zeros(p["mem_block_size"])

    def reset_mem_block(self):
        """
        Clears the memory block of any stored information
        :return:
        """
        self.mem_block = np.zeros(self.mem_block_size)

    def update_memory(self, nn_wgate, nn_encoded_mem):
        """
        GRU-MB agent updates the stored memory
        :param nn_wgate:
        :param nn_encoded_mem:
        :return:
        """
        alpha = 0.1
        wgate = np.reshape(nn_wgate, [1, self.mem_block_size])
        enc_mem = np.reshape(nn_encoded_mem, [1, self.mem_block_size])

        var1 = (1-alpha)*(self.mem_block + np.multiply(wgate, enc_mem))
        var2 = alpha*(np.multiply(wgate, enc_mem) + np.multiply((1-wgate), self.mem_block))

        self.mem_block = var1 + var2


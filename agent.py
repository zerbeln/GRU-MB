import numpy as np

class Agent:

    def __init__(self, parameters):
        self.mem_block_size = parameters["mem_block_size"]
        self.mem_block = np.zeros(parameters["mem_block_size"])

    def reset_mem_block(self):
        self.mem_block = np.zeros(self.mem_block_size)

    def update_memory(self, nn_wgate, nn_encoded_mem):
        alpha = 0.1
        wgate = np.reshape(nn_wgate, [1, self.mem_block_size])
        enc_mem = np.reshape(nn_encoded_mem, [1, self.mem_block_size])

        var1 = (1-alpha)*(self.mem_block + np.multiply(wgate, enc_mem))
        var2 = alpha*(np.multiply(wgate, enc_mem) + np.multiply((1-wgate), self.mem_block))

        self.mem_block = var1 + var2


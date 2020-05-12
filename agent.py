import numpy as np

class Agent:

    def __init__(self, parameters):
        self.mem_block_size = parameters["mem_block_size"]
        self.mem_block = np.zeros(parameters["mem_block_size"])

    def reset_mem_block(self):
        self.mem_block = np.zeros(self.mem_block_size)

    def update_memory(self, nn_wgate_outputs):
        for i in range(self.mem_block_size):
            self.mem_block[i] = nn_wgate_outputs[0, i]


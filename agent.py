import numpy as np

class Agent:

    def __init__(self, tsteps):
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.mem_block = np.zeros((tsteps, 4))   #[S, A, S', R]

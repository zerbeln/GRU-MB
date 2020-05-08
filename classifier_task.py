import numpy as np
import random


class sequenceClassifier:

    def __init__(self):
        self.depth = 10
        self.string_length = 20
        self.training_set_size = 30
        self.test_set_size = 10
        self.training_set = {}
        self.test_set = {}
        self.training_set_answers = np.zeros((self.training_set_size, 2))
        self.test_set_answers = np.zeros((self.test_set_size, 2))

    def generate_training_set(self):

        self.training_set_answers = np.zeros((self.training_set_size, 2))
        for s in range(self.training_set_size):
            seq = np.zeros(self.string_length)
            counter = 0

            while counter < self.depth:
                rnum = random.uniform(0, 1)
                bit = random.randint(0, self.string_length-1)
                if seq[bit] == 0:
                    if rnum < 0.5:
                        seq[bit] = -1
                        self.training_set_answers[s, 0] += 1
                    else:
                        seq[bit] = 1
                        self.training_set_answers[s, 1] += 1
                    counter += 1

            self.training_set["set(0)".format(s)] = seq

    def generate_test_set(self):

        self.test_set_asnwers = np.zeros((self.test_set_size, 2))
        for s in range(self.test_set_size):
            seq = np.zeros(self.string_length)
            counter = 0

            while counter < self.depth:
                rnum = random.uniform(0, 1)
                bit = random.randint(0, self.string_length - 1)
                if seq[bit] == 0:
                    if rnum < 0.5:
                        seq[bit] = -1
                        self.test_set_answers[s, 0] += 1
                    else:
                        seq[bit] = 1
                        self.test_set_answers[s, 1] += 1
                    counter += 1

            self.test_set["set(0)".format(s)] = seq

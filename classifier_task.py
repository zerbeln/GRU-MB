import numpy as np
import random


class sequenceClassifier:

    def __init__(self, param):
        self.depth = param["depth"]
        self.training_set_size = param["train_set_size"]
        self.test_set_size = param["test_set_size"]
        self.training_set = {}
        self.test_set = {}
        self.training_set_answers = np.zeros((self.training_set_size, 3))
        self.test_set_answers = np.zeros((self.test_set_size, 3))

    def generate_training_set(self):
        """
        Create a training set to train the GRU-MB
        :return:
        """
        self.training_set = {}
        for s in range(self.training_set_size):
            seq = []
            counter = 0
            bit = 0
            while counter < self.depth:
                rnum = random.uniform(0, 1)
                if rnum < 0.33:
                    seq.append(0)
                elif rnum < 0.67:
                    seq.append(-1)
                    self.training_set_answers[s, 0] += 1
                    counter += 1
                else:
                    seq.append(1)
                    self.training_set_answers[s, 1] += 1
                    counter += 1
                bit += 1

            if self.training_set_answers[s, 0] > self.training_set_answers[s, 1]:
                self.training_set_answers[s, 2] = -1
            elif self.training_set_answers[s, 0] < self.training_set_answers[s, 1]:
                self.training_set_answers[s, 2] = 1
            else:
                if seq[len(seq)-1] == 1:
                    seq[len(seq)-1] = -1
                    self.training_set_answers[s, 0] += 1
                    self.training_set_answers[s, 1] -= 1
                    self.training_set_answers[s, 2] = -1
                elif seq[len(seq)-1] == -1:
                    seq[len(seq) - 1] = 1
                    self.training_set_answers[s, 0] -= 1
                    self.training_set_answers[s, 1] += 1
                    self.training_set_answers[s, 2] = 1

            self.training_set["set{0}".format(s)] = seq

    def generate_test_set(self):
        """
        Create a test set of sequences to test the best GRU-MB
        :return:
        """
        self.test_set = {}
        for s in range(self.test_set_size):
            seq = []
            counter = 0
            bit = 0
            while counter < self.depth:
                rnum = random.uniform(0, 1)
                if rnum < 0.33:
                    seq.append(0)
                elif rnum < 0.67:
                    seq.append(-1)
                    self.test_set_answers[s, 0] += 1
                    counter += 1
                else:
                    seq.append(1)
                    self.test_set_answers[s, 1] += 1
                    counter += 1
                bit += 1

            if self.test_set_answers[s, 0] > self.test_set_answers[s, 1]:
                self.test_set_answers[s, 2] = -1
            elif self.test_set_answers[s, 0] < self.test_set_answers[s, 1]:
                self.test_set_answers[s, 2] = 1
            else:
                if seq[len(seq)-1] == 1:
                    seq[len(seq)-1] = -1
                    self.test_set_answers[s, 0] += 1
                    self.test_set_answers[s, 1] -= 1
                    self.test_set_answers[s, 2] = -1
                elif seq[len(seq)-1] == -1:
                    seq[len(seq) - 1] = 1
                    self.test_set_answers[s, 0] -= 1
                    self.test_set_answers[s, 1] += 1
                    self.test_set_answers[s, 2] = 1

            self.test_set["set{0}".format(s)] = seq

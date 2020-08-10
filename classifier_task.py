import numpy as np
import random
import pickle
import os
from parameters import parameters as p

class sequenceClassifier:

    def __init__(self):
        self.depth = p["depth"]
        self.training_set_size = p["train_set_size"]
        self.test_set_size = p["test_set_size"]
        self.training_set = {}
        self.test_set = {}
        self.training_set_answers = np.zeros((self.training_set_size, 3))
        self.test_set_answers = np.zeros((self.test_set_size, 3))

    def save_training_set(self):
        """
        Save training sets as a pickle file for re-use
        :return:
        """
        dir_name = 'Saved_Sequences'

        if not os.path.exists(dir_name):  # If directory does not exist, create it
            os.makedirs(dir_name)

        file_path = os.path.join(dir_name, "Training_Sets")
        training_file = open(file_path, 'wb')
        pickle.dump(self.training_set, training_file)
        training_file.close()

    def load_training_set(self):
        """
        Load pre-created training sets from a pickle file
        :return:
        """
        dir_name = 'Saved_Sequences'
        file_path = os.path.join(dir_name, "Training_Sets")
        training_file = open(file_path, 'rb')
        self.training_set = pickle.load(training_file)
        training_file.close()

    def create_training_set(self):
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

    def save_test_set(self):
        """
        Save test set as a pickle file for re-use
        :return:
        """
        dir_name = 'Saved_Sequences'

        if not os.path.exists(dir_name):  # If directory does not exist, create it
            os.makedirs(dir_name)

        file_path = os.path.join(dir_name, "Test_Set")
        training_file = open(file_path, 'wb')
        pickle.dump(self.training_set, training_file)
        training_file.close()

    def load_test_set(self):
        """
        Load a pre-created test set from a pickle file
        :return:
        """
        dir_name = 'Saved_Sequences'
        file_path = os.path.join(dir_name, "Test_Set")
        training_file = open(file_path, 'rb')
        self.training_set = pickle.load(training_file)
        training_file.close()

    def create_test_set(self):
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

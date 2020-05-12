import numpy as np
import random


class Ccea:

    def __init__(self, parameters):
        self.population = {}
        self.pop_size = int(parameters["pop_size"])
        self.mut_rate = parameters["m_rate"]
        self.mut_chance = parameters["m_prob"]
        self.eps = parameters["epsilon"]
        self.fitness = np.zeros(self.pop_size)
        self.team_selection = np.ones(self.pop_size) * (-1)
        self.n_elites = parameters["n_elites"]  # Number of elites selected from each gen

        # Numbers of weights for GRU-MB
        self.n_layer1_w = (parameters["n_outputs"]+parameters["n_inputs"]+1)*parameters["n_hnodes"]
        self.n_layer2_w = (parameters["n_hnodes"]+1)*parameters["n_outputs"]
        self.n_igate_w = (parameters["n_inputs"]+1)*parameters["n_inputs"]
        self.n_rgate_w = (parameters["mem_block_size"]+1)*parameters["n_outputs"]
        self.n_wgate_w = (parameters["n_outputs"]+1)*parameters["mem_block_size"]

        policy = {}
        for pop_id in range(self.pop_size):
            policy["layer1_weights"] = np.random.rand(self.n_layer1_w)
            policy["layer2_weights"] = np.random.rand(self.n_layer2_w)
            policy["igate_weights"] = np.random.rand(self.n_igate_w)
            policy["rgate_weights"] = np.random.rand(self.n_rgate_w)
            policy["wgate_weights"] = np.random.rand(self.n_wgate_w)

            self.population["pop(0)".format(pop_id)] = policy

    def reset_population(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        :return: None
        """

        self.population = {}
        self.fitness = np.zeros(self.pop_size)
        self.team_selection = np.ones(self.pop_size) * (-1)

        policy = {}
        for pop_id in range(self.pop_size):
            policy["layer1_weights"] = np.random.rand(self.n_layer1_w)
            policy["layer2_weights"] = np.random.rand(self.n_layer2_w)
            policy["igate_weights"] = np.random.rand(self.n_igate_w)
            policy["rgate_weights"] = np.random.rand(self.n_rgate_w)
            policy["wgate_weights"] = np.random.rand(self.n_wgate_w)

            self.population["pop(0)".format(pop_id)] = policy


    def select_policy_teams(self):  # Create policy teams for testing
        """
        Choose teams of individuals from among populations to be tested
        :return: None
        """

        self.team_selection = np.ones(self.pop_size) * (-1)

        for policy_id in range(self.pop_size):
            target = random.randint(0, (self.pop_size - 1))  # Select a random policy from pop
            k = 0
            while k < policy_id:  # Check for duplicates
                if target == self.team_selection[k]:
                    target = random.randint(0, (self.pop_size - 1))
                    k = -1
                k += 1
            self.team_selection[policy_id] = target  # Assign policy to team

    def weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        :return:
        """

        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            # Layer 1 Mutation
            for w in range(self.n_layer1_w):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["layer1_weights"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop(0)".format(starting_pol)]["layer1_weights"][w] += mutation

            # Layer 2 Mutation
            for w in range(self.n_layer2_w):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["layer2_weights"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["layer2_weights"][w] += mutation

            # Input Gate Mutation
            for w in range(self.n_igate_w):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["igate_weights"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["igate_weights"][w] += mutation

            # Read Gate Mutation
            for w in range(self.n_rgate_w):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["rgate_weights"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["rgate_weights"][w] += mutation

            # Write Gate Mutation
            for w in range(self.n_wgate_w):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["wgate_weights"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["wgate_weights"][w] += mutation

            starting_pol += 1

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents from which an offspring population will be created
        :return: None
        """

        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                max_index = np.argmax(self.fitness)
                new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(max_index)]
            else:
                rnum = random.uniform(0, 1)
                if rnum > self.eps:
                    max_index = np.argmax(self.fitness)
                    new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(max_index)]
                else:
                    parent = random.randint(1, (self.pop_size-1))
                    new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(parent)]

        self.population = new_population

    def fitness_prop_selection(self):
        summed_fitness = np.sum(self.fitness)
        fit_brackets = np.zeros(self.pop_size)

        for pol_id in range(self.pop_size):
            if pol_id == 0:
                fit_brackets[pol_id] = self.fitness[pol_id]/summed_fitness
            else:
                fit_brackets[pol_id] = fit_brackets[pol_id-1] + self.fitness[pol_id]/summed_fitness

        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                max_index = np.argmax(self.fitness)
                new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(max_index)]
            else:
                rnum = random.uniform(0, 1)
                for p_id in range(self.pop_size):
                    if p_id == 0 and rnum < fit_brackets[0]:
                        new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(0)]
                    elif fit_brackets[p_id-1] <= rnum < fit_brackets[p_id]:
                        new_population["pop(0)".format(pol_id)] = self.population["pop(0)".format(p_id)]

        self.population = new_population

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        :return: None
        """
        # self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        self.fitness_prop_selection()  # Select k successors using fit prop selection
        self.weight_mutate()  # Mutate successors

    def reset_fitness(self):
        self.fitness = np.zeros(self.pop_size)

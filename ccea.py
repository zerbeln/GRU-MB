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

        # Numbers of weights for GRU-MB
        self.n_layer1_w = (parameters["mem_block_size"]+parameters["n_inputs"]+1)*parameters["n_hnodes"]
        self.n_layer2_w = (parameters["n_hnodes"]+1)*parameters["n_outputs"]
        self.n_igate_w = (parameters["n_inputs"]+1)*parameters["n_inputs"]
        self.n_rgate_w = (parameters["mem_block_size"]+1)*parameters["mem_block_size"]
        self.n_wgate_w = (parameters["mem_block_size"]+1)*parameters["mem_block_size"]

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

        for pol_id in range(self.offspring_psize):
            for w in range(self.policy_size):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mutation = (np.random.normal(0, self.mut_rate) * self.offspring_pop[pol_id, w])
                    self.offspring_pop[pol_id, w] += mutation

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents from which an offspring population will be created
        :return: None
        """

        policy_id = 0
        while policy_id < self.parent_psize:
            rnum = random.uniform(0, 1)
            if rnum > self.eps:  # Choose best policy
                pol_index = np.argmax(self.fitness)
                self.parent_pop[policy_id] = self.pops[pol_index].copy()
            else:
                parent = random.randint(1, (self.total_pop_size - 1))  # Choose a random parent
                self.parent_pop[policy_id] = self.pops[parent].copy()
            policy_id += 1

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        :return: None
        """
        self.rank_individuals()
        self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        self.offspring_pop = self.parent_pop.copy()
        self.weight_mutate()  # Mutate successors
        self.combine_pops()

    def rank_individuals(self):
        """
        Order individuals in the population based on their fitness scores
        :return: None
        """

        for pol_id_a in range(self.pop_size - 1):
            pol_id_b = pol_id_a + 1
            while pol_id_b < (self.pop_size):
                if pol_id_a != pol_id_b:
                    if self.fitness[pol_id_a] < self.fitness[pol_id_b]:
                        self.fitness[pol_id_a], self.fitness[pol_id_b] = self.fitness[pol_id_b], self.fitness[pol_id_a]
                        self.pops[pol_id_a], self.pops[pol_id_b] = self.pops[pol_id_b], self.pops[pol_id_a]
                pol_id_b += 1

    def reset_fitness(self):
        self.fitness = np.zeros(self.pop_size)
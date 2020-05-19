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

        # Network parameters
        self.n_inputs = parameters["n_inputs"]
        self.n_hnodes = parameters["n_hnodes"]
        self.n_outputs = parameters["n_outputs"]
        self.mem_block_size = parameters["mem_block_size"]

        policy = {}
        for pop_id in range(self.pop_size):
            policy["b_out"] = np.random.rand(self.n_outputs)
            policy["p_out"] = np.random.rand(self.mem_block_size)

            # Input Gate
            policy["k_igate"] = np.random.rand(self.mem_block_size)
            policy["r_igate"] = np.random.rand(self.mem_block_size)
            policy["n_igate"] = np.random.rand(self.mem_block_size**2)
            policy["b_igate"] = np.random.rand(self.mem_block_size)

            # Block Input
            policy["k_block"] = np.random.rand(self.mem_block_size)
            policy["n_block"] = np.random.rand(self.mem_block_size**2)
            policy["b_block"] = np.random.rand(self.mem_block_size)

            # Read Gate
            policy["k_rgate"] = np.random.rand(self.mem_block_size)
            policy["r_rgate"] = np.random.rand(self.mem_block_size)
            policy["n_rgate"] = np.random.rand(self.mem_block_size**2)
            policy["b_rgate"] = np.random.rand(self.mem_block_size)

            # Write Gate
            policy["k_wgate"] = np.random.rand(self.mem_block_size)
            policy["r_wgate"] = np.random.rand(self.mem_block_size)
            policy["n_wgate"] = np.random.rand(self.mem_block_size**2)
            policy["b_wgate"] = np.random.rand(self.mem_block_size)

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
            policy["b_out"] = np.random.rand(self.n_outputs)
            policy["p_out"] = np.random.rand(self.mem_block_size)

            # Input Gate
            policy["k_igate"] = np.random.rand(self.mem_block_size)
            policy["r_igate"] = np.random.rand(self.mem_block_size)
            policy["n_igate"] = np.random.rand(self.mem_block_size ** 2)
            policy["b_igate"] = np.random.rand(self.mem_block_size)

            # Block Input
            policy["k_block"] = np.random.rand(self.mem_block_size)
            policy["n_block"] = np.random.rand(self.mem_block_size ** 2)
            policy["b_block"] = np.random.rand(self.mem_block_size)

            # Read Gate
            policy["k_rgate"] = np.random.rand(self.mem_block_size)
            policy["r_rgate"] = np.random.rand(self.mem_block_size)
            policy["n_rgate"] = np.random.rand(self.mem_block_size ** 2)
            policy["b_rgate"] = np.random.rand(self.mem_block_size)

            # Write Gate
            policy["k_wgate"] = np.random.rand(self.mem_block_size)
            policy["r_wgate"] = np.random.rand(self.mem_block_size)
            policy["n_wgate"] = np.random.rand(self.mem_block_size ** 2)
            policy["b_wgate"] = np.random.rand(self.mem_block_size)

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

    def mutate_igate(self):
        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            for w in range(self.mem_block_size):
                # Bias Weights
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["b_igate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["b_igate"][w] += mutation

                # K Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["k_igate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["k_igate"][w] += mutation

                # R Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["r_igate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["r_igate"][w] += mutation

            for w in range(self.mem_block_size ** 2):
                # N Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["n_igate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["n_igate"][w] += mutation

            starting_pol += 1

    def mutate_rgate(self):
        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            for w in range(self.mem_block_size):
                # Bias Weights
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["b_rgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["b_rgate"][w] += mutation

                # K Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["k_rgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["k_rgate"][w] += mutation

                # R Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["r_rgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["r_rgate"][w] += mutation

            for w in range(self.mem_block_size ** 2):
                # N Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["n_rgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["n_rgate"][w] += mutation

            starting_pol += 1

    def mutate_wgate(self):
        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            for w in range(self.mem_block_size):
                # Bias Weights
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["b_wgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["b_wgate"][w] += mutation

                # K Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["k_wgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["k_wgate"][w] += mutation

                # R Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["r_wgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["r_wgate"][w] += mutation

            for w in range(self.mem_block_size ** 2):
                # N Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["n_wgate"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["n_wgate"][w] += mutation

            starting_pol += 1

    def mutate_block(self):
        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            for w in range(self.mem_block_size):
                # Bias Weights
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["b_block"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["b_block"][w] += mutation

                # K Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["k_block"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["k_block"][w] += mutation

            for w in range(self.mem_block_size ** 2):
                # N Matrix
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["n_block"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["n_block"][w] += mutation

            starting_pol += 1

    def weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        :return:
        """

        starting_pol = int(self.n_elites)
        while starting_pol < self.pop_size:
            # Layer 1 Mutation
            for w in range(self.n_outputs):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["b_out"][w]
                    mutation = np.random.normal(0, self.mut_rate) * weight
                    self.population["pop(0)".format(starting_pol)]["b_out"][w] += mutation

            # Layer 2 Mutation
            for w in range(self.mem_block_size):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    weight = self.population["pop(0)".format(starting_pol)]["p_out"][w]
                    mutation = (np.random.normal(0, self.mut_rate)) * weight
                    self.population["pop(0)".format(starting_pol)]["p_out"][w] += mutation

            starting_pol += 1

        self.mutate_igate()
        self.mutate_rgate()
        self.mutate_wgate()

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

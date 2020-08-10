parameters = {}

# Test Parameters
parameters["s_runs"] = 1
parameters["create_new_sets"] = 1  # 1 = create new sequences, 0 = re-use sequences

# Sequence Classifier
parameters["depth"] = 4
parameters["train_set_size"] = 50
parameters["test_set_size"] = 20

# Neural Network Parameters
parameters["n_inputs"] = 1
parameters["n_hnodes"] = 5
parameters["n_outputs"] = 1
parameters["mem_block_size"] = 5  # Number of elements in each memory block entry

# CCEA Parameters
parameters["pop_size"] = 100
parameters["m_rate"] = 0.1
parameters["m_prob"] = 0.1
parameters["epsilon"] = 0.1
parameters["generations"] = 1000
parameters["n_elites"] = 10

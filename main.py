from neural_net import NeuralNetwork
import numpy as np
from ccea import Ccea
from agent import Agent
from classifier_task import sequenceClassifier
import csv; import os


def set_parameters():
    """
    Create a dictionary of all critical test parameters
    :return:
    """
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

    return parameters


def save_reward_history(reward_history, file_name):
    """
    Save the reward history for the best policy using the test set at each generation
    :param reward_history:
    :param file_name:
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def main():
    # Test Parameters
    p = set_parameters()
    sc = sequenceClassifier(p)
    ag = Agent(p)
    cc = Ccea(p)
    nn = NeuralNetwork(p)

    if p["create_new_sets"] == 1:
        sc.create_training_set()
        sc.save_training_set()
        sc.create_test_set()
        sc.save_test_set()
    else:
        sc.load_training_set()
        sc.load_test_set()

    for s in range(p["s_runs"]):
        print("Stat Run: ", s)
        # Training
        training_reward_history = []
        test_reward_history = []
        state_vec = np.ones(p["n_inputs"])

        cc.create_new_population()

        for gen in range(p["generations"]):
            print("Gen: ", gen)
            pop_id = cc.n_elites
            while pop_id < p["pop_size"]:  # Test each set of weights in EA
                nn.reset_nn()
                nn.get_weights(cc.population["pop{0}".format(pop_id)])
                fitness_score = 0.0

                for seq in range(p["train_set_size"]):
                    ag.reset_mem_block()
                    nn.clear_outputs()
                    seq_len = len(sc.training_set["set{0}".format(seq)])
                    current_sequence = sc.training_set["set{0}".format(seq)].copy()

                    for num in range(seq_len):
                        state_vec[0] = current_sequence[num]
                        nn.run_neural_network(state_vec, ag.mem_block)
                        ag.update_memory(nn.wgate_outputs, nn.encoded_memory)

                    if nn.out_layer[0] < 0.5 and sc.training_set_answers[seq, 2] == -1:
                        fitness_score += 1
                    elif nn.out_layer[0] >= 0.5 and sc.training_set_answers[seq, 2] == 1:
                        fitness_score += 1

                cc.fitness[pop_id] = fitness_score/p["train_set_size"]
                pop_id += 1

            # Testing
            nn.reset_nn()
            state_vec = np.ones(p["n_inputs"])
            best_pol_id = np.argmax(cc.fitness)  # Find the best policy in the population currently
            nn.get_weights(cc.population["pop{0}".format(best_pol_id)])
            test_reward = 0.0

            for seq in range(p["test_set_size"]):
                ag.reset_mem_block()
                nn.clear_outputs()
                seq_len = len(sc.test_set["set{0}".format(seq)])
                current_sequence = sc.test_set["set{0}".format(seq)].copy()

                for num in range(seq_len):
                    state_vec[0] = current_sequence[num]
                    nn.run_neural_network(state_vec, ag.mem_block)
                    ag.update_memory(nn.block_output, nn.wgate_outputs)

                if nn.out_layer[0] < 0.5 and sc.training_set_answers[seq, 2] == -1:
                    test_reward += 1
                elif nn.out_layer[0] >= 0.5 and sc.training_set_answers[seq, 2] == 1:
                    test_reward += 1

            test_reward_history.append(test_reward/p["test_set_size"])
            training_reward_history.append(max(cc.fitness))
            cc.down_select()

        save_reward_history(training_reward_history, "Training_Fitness.csv")
        save_reward_history(test_reward_history, "Test_Reward.csv")


main()

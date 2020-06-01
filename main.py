from neural_net import NeuralNetwork
import numpy as np
from ccea import Ccea
from agent import Agent
from classifier_task import sequenceClassifier
import csv; import os

def set_parameters():
    parameters = {}

    # Test Parameters
    parameters["n_agents"] = 1
    parameters["s_runs"] = 1

    # Sequence Classifier
    parameters["depth"] = 4
    parameters["train_set_size"] = 200
    parameters["test_set_size"] = 50

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
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)

def main():
    # Test Parameters
    param = set_parameters()
    sc = sequenceClassifier(param)
    ag = Agent(param)
    cc = Ccea(param)
    nn = NeuralNetwork(param)

    for s in range(param["s_runs"]):
        print("Stat Run: ", s)
        # Training
        training_reward_history = []
        test_reward_history = []
        sc.generate_training_set()
        state_vec = np.ones(param["n_inputs"])
        if s > 0:
            cc.reset_population()
        for gen in range(param["generations"]):
            pop_id = cc.n_elites
            while pop_id < param["pop_size"]:  # Test each set of weights in EA
                nn.reset_nn()
                nn.get_weights(cc.population["pop{0}".format(pop_id)])
                fitness_score = 0.0

                for seq in range(param["train_set_size"]):
                    ag.reset_mem_block()
                    nn.clear_outputs()
                    seq_len = len(sc.training_set["set{0}".format(seq)])
                    current_sequence = sc.training_set["set{0}".format(seq)].copy()

                    for num in range(seq_len):
                        state_vec[0] = current_sequence[num]
                        nn.run_neural_network(state_vec, ag.mem_block)
                        ag.update_memory(nn.block_output, nn.wgate_outputs)

                    if nn.out_layer[0] < 0.5 and sc.training_set_answers[seq, 2] == -1:
                        fitness_score += 1
                    elif nn.out_layer[0] >= 0.5 and sc.training_set_answers[seq, 2] == 1:
                        fitness_score += 1

                cc.fitness[pop_id] = fitness_score
                pop_id += 1

            # Testing
            sc.generate_test_set()
            state_vec = np.ones(param["n_inputs"])
            best_pol_id = np.argmax(cc.fitness)
            nn.reset_nn()
            nn.get_weights(cc.population["pop{0}".format(best_pol_id)])
            test_reward = 0.0

            for seq in range(param["test_set_size"]):
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

            test_reward_history.append(test_reward)
            training_reward_history.append(max(cc.fitness))
            cc.down_select()

        save_reward_history(training_reward_history, "Training_Fitness.csv")
        save_reward_history(test_reward_history, "Test_Reward.csv")


main()

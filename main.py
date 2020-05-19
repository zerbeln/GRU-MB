from neural_net import NeuralNetwork
import numpy as np
from ccea import Ccea
from agent import Agent
from classifier_task import sequenceClassifier

def set_parameters():
    parameters = {}

    # Test Parameters
    parameters["n_agents"] = 1
    parameters["s_runs"] = 1

    # Sequence Classifier
    parameters["depth"] = 5
    parameters["train_set_size"] = 50
    parameters["test_set_size"] = 10

    # Neural Network Parameters
    parameters["n_inputs"] = 1
    parameters["n_hnodes"] = 5
    parameters["n_outputs"] = 1
    parameters["mem_block_size"] = 5  # Number of elements in each memory block entry

    # CCEA Parameters
    parameters["pop_size"] = 30
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.1
    parameters["epsilon"] = 0.1
    parameters["generations"] = 500
    parameters["n_elites"] = 2

    return parameters


def main():
    # Test Parameters
    param = set_parameters()

    for s in range(param["s_runs"]):
        state_vec = np.ones(param["n_inputs"])

        sc = sequenceClassifier(param)
        ag = Agent(param)
        cc = Ccea(param)
        nn = NeuralNetwork(param)

        # Training
        sc.generate_training_set()
        for gen in range(param["generations"]):
            for pol in range(param["pop_size"]):  # Test each set of weights in EA
                nn.reset_nn()
                weights = cc.population["pop(0)".format(pol)]
                nn.get_weights(weights)

                fitness_score = 0.0
                for seq in range(param["train_set_size"]):
                    ag.reset_mem_block()
                    seq_len = len(sc.training_set["set(0)".format(seq)])
                    for num in range(seq_len):
                        state_vec[0] = sc.training_set["set(0)".format(seq)][num]
                        nn.run_neural_network(state_vec, ag.mem_block)
                        ag.update_memory(nn.block_output, nn.wgate_outputs)

                    if nn.out_layer[0] < 0.5 and sc.training_set_answers[seq, 2] == -1:
                        fitness_score += 1
                    elif nn.out_layer[0] >= 0.5 and sc.training_set_answers[seq, 2] == 1:
                        fitness_score += 1

                cc.fitness[pol] = fitness_score

            cc.down_select()
            print(max(cc.fitness))


main()

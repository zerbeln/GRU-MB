from neural_net import NeuralNetwork
import numpy as np
from ccea import Ccea
from agent import Agent

def set_parameters():
    parameters = {}

    # Test Parameters
    parameters["n_agents"] = 1
    parameters["t_steps"] = 1
    parameters["s_runs"] = 1

    # Neural Network Parameters
    parameters["n_inputs"] = 8
    parameters["n_hnodes"] = 10
    parameters["n_outputs"] = 2
    parameters["mem_block_size"] = 4  # Number of elements in each memory block entry

    # CCEA Parameters
    parameters["pop_size"] = 10
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.1
    parameters["epsilon"] = 0.1
    parameters["generations"] = 1

    return parameters

def main():

    # Test Parameters
    param = set_parameters()

    # Quick Test for GRU-MB architecture
    state_vec = np.ones(param["n_inputs"])

    agents = {}  # Dictionary for containing agents

    for ag_id in range(param["n_agents"]):
        agents["Agent(0)".format(ag_id)] = Agent(param["t_steps"])
        agents["CCEA(0)".format(ag_id)] = Ccea(param)
        agents["NN(0)".format(ag_id)] = NeuralNetwork(param)

    for gen in range(param["generations"]):
        for ag_id in range(param["n_agents"]):
            agents["CCEA(0)".format(ag_id)].select_policy_teams()

        for team in range(param["pop_size"]):

            for ag_id in range(param["n_agents"]):
                agents["NN(0)".format(ag_id)].reset_nn()
                pol_id = agents["CCEA(0)".format(ag_id)].team_selection[team]
                weights = agents["CCEA(0)".format(ag_id)].population["pop(0)".format(pol_id)]
                agents["NN(0)".format(ag_id)].get_weights(weights)

            for tstep in range(param["t_steps"]):
                for ag_id in range(param["n_agents"]):
                    agents["NN(0)".format(ag_id)].run_neural_network(tstep, state_vec)


main()

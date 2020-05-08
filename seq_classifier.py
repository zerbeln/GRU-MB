import numpy as np, os
import mod_mmu as mod, sys
from random import randint
from torch.autograd import Variable
import torch
import random
from torch.utils import data as util


class Tracker:  # Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 10: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')


class Parameters:
    def __init__(self):
            #BackProp
            self.total_epochs= 1000
            self.batch_size = 100
            self.train_size = 10000
            #Determine the nerual archiecture
            self.arch_type = 2 #1 MMU-Add
                               #2 MMU

            #Task Params
            self.depth_train = 10
            self.corridors = [1, 1]
            self.output_activation = 'sigmoid'

            # Auto
            self.num_input = 1
            self.num_hidden = 25
            self.num_output = 1
            self.num_memory = self.num_hidden
            if self.arch_type == 1:
                self.arch_type = 'MMU_Add'
            elif self.arch_type == 2:
                self.arch_type = 'MMU'
            else:
                sys.exit('Invalid choice of neural architecture')
            self.save_foldername = 'Seq_Classifier/'


class TaskSeqClassifier:  # Sequence Classifier
    def __init__(self, parameters):
        self.parameters = parameters
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        if self.parameters.arch_type == 'MMU_Add':
            mem_add = True
        elif self.parameters.arch_type == 'MMU':
            mem_add = False
        self.model = mod.PT_Net(parameters.num_input, parameters.num_hidden, parameters.num_memory, parameters.num_output, mem_add)




    def load(self, filename):
        return torch.load(self.save_foldername + filename)

    def run_bprop(self, all_train_x, all_train_y):

        #criterion = torch.nn.L1Loss(False)
        criterion = torch.nn.SmoothL1Loss(False)
        #criterion = torch.nn.KLDivLoss()
        #criterion = torch.nn.MSELoss()
        #criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.1)
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.5, nesterov = True)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005, momentum=0.1)


        #Data process
        self.pad_data(all_train_x, all_train_y)
        seq_len = len(all_train_x[0])

        #Is relevant markers to reduce backprop
        relevance_mat = []
        for i in range(seq_len):
            inp = np.array(all_train_x)[:, i]
            is_relevant = (inp == 1) + (inp == -1)
            if is_relevant.any(): relevance_mat.append(True)
            else: relevance_mat.append(False)


        eval_train_y = all_train_y[:] # Copy just the list to evaluate batch
        all_train_x = torch.Tensor(all_train_x).cuda()
        all_train_y = torch.Tensor(all_train_y).cuda()
        eval_train_x = all_train_x[:]  # Copy tensor to evaluate batch
        train_dataset = util.TensorDataset(all_train_x, all_train_y)
        train_loader = util.DataLoader(train_dataset, batch_size=self.parameters.batch_size, shuffle=True)

        self.model.cuda()
        for epoch in range(1, self.parameters.total_epochs+1):

            epoch_loss = 0.0
            for data in train_loader: #Each Batch
                net_inputs, targets = data
                self.model.reset(self.parameters.batch_size)  # Reset memory and recurrent out for the model
                for i in range(seq_len):  # For the length of the sequence
                    net_inp = Variable(net_inputs[:,i], requires_grad=True).unsqueeze(0)

                    net_out = self.model.forward(net_inp)
                    target_T = Variable(targets[:,i]).unsqueeze(0)
                    loss = criterion(net_out, target_T)
                    if relevance_mat[i]: loss.backward(retain_graph=True)
                    epoch_loss += loss.cpu().data.numpy()

            optimizer.step()  # Perform the gradient updates to weights for the entire set of collected gradients
            optimizer.zero_grad()

            if epoch % 10 == 0:
                test_x, test_y = self.generate_task_seq(50, parameters.depth_train)
                self.pad_data(test_x, test_y)
                test_x = torch.Tensor(test_x).cuda()
                train_fitness = self.batch_evaluate(self.model, eval_train_x, eval_train_y)
                valid_fitness = self.batch_evaluate(self.model, test_x, test_y)
                print( 'Epoch: ', epoch, ' Loss: ',  epoch_loss,' Train_Performance:', "%0.2f" % train_fitness,' Valid_Performance:', "%0.2f" % valid_fitness)
                tracker.update([epoch_loss, train_fitness, valid_fitness], epoch)
                torch.save(self.model, self.save_foldername + 'seq_classifier_net')




    def batch_evaluate(self, model, test_x, test_y):
        seq_len = len(test_x[0])
        model.reset(len(test_x))  # Reset memory and recurrent out for the model
        test_failure = np.zeros((1, len(test_x))).astype('bool') #Track failure of test

        for i in range(seq_len):  # For the length of the sequence
            net_inp = Variable(test_x[:,i], requires_grad=True).unsqueeze(0)
            net_out = model.forward(net_inp).cpu().data.numpy()
            target = np.reshape(np.array(test_y)[:, i], (1, len(test_x)))

            inp = test_x[:, i].unsqueeze(0).cpu().numpy()
            is_relevant = (inp == 1) + (inp == -1) #
            net_out_bool = (net_out >= 0.5)
            is_incorrect = np.logical_xor(net_out_bool, target)

            test_failure = (test_failure + (is_relevant * is_incorrect))

        test_failure = (test_failure > 0)
        return (1.0 - np.sum(test_failure)/float(len(test_x)))

    def generate_task_seq(self, num_examples, depth):
        train_x = []; train_y = []
        for example in range(num_examples):
            x = []; y = []
            for i in range(depth):
                #Encode the signal (1 or -1s)
                if random.random() < 0.5: x.append(-1)
                else: x.append(1)
                if sum(x) >= 0: y.append(1)
                else: y.append(0)
                if i == depth - 1: continue

                #Encdoe the noise (0's)
                num_noise = randint(self.parameters.corridors[0], self.parameters.corridors[1])
                for i in range(num_noise):
                    x.append(0); y.append(y[-1])
            train_x.append(x); train_y.append(y)
        return train_x, train_y


    def pad_data(self, all_train_x, all_train_y):
        #Pad train_data
        all_len = [len(train_x) for train_x in all_train_x]
        max_len = max(all_len)
        for i in range(len(all_train_x)):
            while len(all_train_x[i]) < max_len:
                all_train_x[i].append(0)
                all_train_y[i].append(all_train_y[i][-1])

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters, ['epoch_loss', 'train', 'valid'], '_seq_classifier.csv')
    print('Running Backprop with', parameters.arch_type)
    sim_task = Task_Seq_Classifier(parameters)

    #Run Back_prop
    train_x, train_y = sim_task.generate_task_seq(parameters.train_size, parameters.depth_train)
    sim_task.run_bprop(train_x, train_y)
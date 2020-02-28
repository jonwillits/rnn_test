import torch
import torch.nn as nn
import pickle
import sys
import numpy as np


class SRN(nn.Module):
    ############################################################################################################
    def __init__(self):

        super(SRN, self).__init__()
        self.net_name = None
        self.training_set = None

        self.optimizer = None
        self.hidden_size = None
        self.learning_rate = None
        self.weight_init = None

        self.input_size = None
        self.output_size = None
        self.current_epoch = None

        self.h_x = None     # weights from input to hidden
        self.y_h = None     # weights from hidden to output
        self.h_h = None     # weights from hidden back to itself
        self.last_h = None
        self.sigmoid = None
        self.criterion = None
        self.criterion2 = None
        self.hidden_act_function = None

        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid().float()
        self.tanh = nn.Tanh().float()
        self.relu = nn.ReLU().float()
        self.softmax = nn.Softmax()
    #
    def init_model(self, training_set, hidden_size, hidden_act_function, learning_rate, weight_init):

        self.training_set = training_set
        self.hidden_size = hidden_size
        self.hidden_act_function = hidden_act_function
        self.learning_rate = learning_rate
        self.weight_init = weight_init

        self.input_size = training_set.num_word_types
        self.output_size = training_set.num_word_types

        self.current_epoch = 0
        h_vector = np.zeros([self.hidden_size], float)
        self.last_h = torch.from_numpy(h_vector)


        self.h_x = nn.Linear(self.input_size, self.hidden_size).float()
        self.y_h = nn.Linear(self.hidden_size, self.output_size).float()
        self.h_h = nn.Linear(self.hidden_size, self.hidden_size).float()

        self.h_x.apply(self.init_weights)
        self.y_h.apply(self.init_weights)
        self.h_h.apply(self.init_weights)

    def forward_item(self, x):
        z_h = self.h_x(x.float()) + self.h_h(self.last_h.float())
        if self.hidden_act_function == 'tanh':
            h = self.tanh(z_h)
        elif self.hidden_actf == 'sigmoid':
            h = self.sigmoid(z_h)
        elif self.hidden_actf == 'relu':
            h = self.relu(z_h)
        else:
            print("ERROR: Improper hidden activation function")
            raise RuntimeError
        z_o = self.y_h(h)
        o = self.sigmoid(z_o)
        self.last_h = h.clone()
        return o, h

    # def test_item(self, x, y):
    #     out, h = self.forward_item(x)
    #     loss = self.criterion2(out.float(), y.float())
    #     return out, h, loss
    #
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.uniform_(-self.weight_init, self.weight_init)
            m.bias.data.uniform_(-self.weight_init, self.weight_init)
        else:
            print("Not a linear weight being initialized")
            sys.exit(0)

    def train_item(self, x, y, torch_optimizer):
        o, h = self.forward_item(x)
        loss = self.criterion(o.float(), y.float())
        torch_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch_optimizer.step()
        return o, loss
    #
    # def create_network_directory(self):
    #     try:
    #         print("Creating models directory")
    #         os.mkdir('models')
    #     except Exception as e:
    #         pass
    #
    #     if self.x_type == 'WorldState':
    #         x_type = "WS"
    #     elif self.x_type == 'HiddenState':
    #         x_type = 'HS'
    #     else:
    #         print("X Type not recognized in directory creation")
    #         sys.exit()
    #
    #     if self.y_type == 'WorldState':
    #         y_type = "WS"
    #     elif self.y_type == 'FeatureVector':
    #         y_type = 'FV'
    #     else:
    #         print("Y Type not recognized in directory creation")
    #         sys.exit()
    #
    #     self.net_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(x_type, y_type,
    #                                                      self.start_datetime[0],
    #                                                      self.start_datetime[1],
    #                                                      self.start_datetime[2],
    #                                                      self.start_datetime[3],
    #                                                      self.start_datetime[4],
    #                                                      self.start_datetime[5],
    #                                                      self.start_datetime[6])
    #     try:
    #         os.mkdir("models/" + self.net_name)
    #     except Exception as e:
    #         print(e)
    #         sys.exit()
    #
    #     header_string = "epoch,time,training_cost,test_cost"
    #     for i in range(self.training_set.num_included_feature_types):
    #         header_string += ',{}_training_cost'.format(self.training_set.included_feature_type_list[i])
    #     for i in range(self.training_set.num_included_feature_types):
    #         header_string += ',{}_test_cost'.format(self.training_set.included_feature_type_list[i])
    #     for i in range(self.training_set.num_included_feature_types):
    #         header_string += ',{}_training_accuracy'.format(self.training_set.included_feature_type_list[i])
    #     for i in range(self.training_set.num_included_feature_types):
    #         header_string += ',{}_test_accuracy'.format(self.training_set.included_feature_type_list[i])
    #     for i in range(self.training_set.num_included_features):
    #         header_string += ',{}_training_accuracy'.format(self.training_set.included_feature_list[i])
    #     for i in range(self.training_set.num_included_features):
    #         header_string += ',{}_test_accuracy'.format(self.training_set.included_feature_list[i])
    #
    #     f = open("models/" + self.net_name + "/performance.csv", 'w')
    #     f.write(header_string + "\n")
    #     f.close()
    #
    #     self.save_network_properties()
    #
    # def save_network_properties(self):
    #     file_location = "models/" + self.net_name + "/network_properties.csv"
    #     f = open(file_location, 'w')
    #     f.write("network_name: {}\n".format(self.net_name))
    #     f.write("x_type: {}\n".format(self.x_type))
    #     f.write("y_type: {}\n".format(self.y_type))
    #     f.write("input_size: {}\n".format(self.input_size))
    #     f.write("hidden_size: {}\n".format(self.hidden_size))
    #     f.write("hidden_actf: {}\n".format(self.hidden_actf))
    #     f.write("output_size: {}\n".format(self.output_size))
    #     f.write("optimizer: {}\n".format(self.optimizer))
    #     f.write("learning_rate: {}\n".format(self.learning_rate))
    #     f.write("weight_init: {}\n".format(self.weight_init))
    #     f.write("training_set: {}\n".format(self.training_set.world_state_filename))
    #     f.write("current_epoch: {}\n".format(self.current_epoch))
    #     f.write("training_time: {}".format(self.training_time))
    #     f.close()
    #
    # def save_network_states(self, dataset):
    #     network_state_list = []
    #     dataset.create_xy(self, False, False)
    #     for i in range(len(dataset.x)):
    #         o, h, o_cost = self.test_item(dataset.x[i], dataset.y[i])
    #         network_state_list.append((dataset.x[i], dataset.y[i], o.detach().cpu().numpy(), h.detach().cpu().numpy()))
    #     file_location = "models/" + self.net_name + "/states.csv".format(self.current_epoch)
    #     outfile = open(file_location, 'wb')
    #     pickle.dump(network_state_list, outfile)
    #     outfile.close()
    #
    # def save_network_weights(self):
    #     file_location = "models/" + self.net_name + "/weights.csv".format(self.current_epoch)
    #     outfile = open(file_location, 'wb')
    #     weights_list = [self.h_x, self.y_h]
    #     pickle.dump(weights_list, outfile)
    #     outfile.close()
    #
    # def save_network_performance(self):
    #     file_location = "models/" + self.net_name + "/performance.csv"
    #     output_string = ""
    #     for item in self.performance_list:
    #         output_string += str(item) + ","
    #     output_string = output_string[:-1]
    #     f = open(file_location, 'a')
    #     f.write(output_string + "\n")
    #     f.close()
import torch
import torch.nn as nn
import datetime
import os
import pickle


class SRN(nn.Module):
    ############################################################################################################
    def __init__(self):

        super(SRN, self).__init__()
        self.name = None
        self.start_datetime = None
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
        self.last_h = None
        self.sigmoid = None
        self.criterion = None
        self.criterion2 = None
        self.hidden_act_function = None

        self.criterion = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid().float()
        self.tanh = nn.Tanh().float()
        self.soft_max = nn.Softmax()
    #
    def init_model(self, training_set, hidden_size, learning_rate, weight_init):

        self.training_set = training_set
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_init = weight_init

        self.input_size = training_set.vocab_size
        self.output_size = training_set.vocab_size

        self.current_epoch = 0

        self.h_x = nn.Linear(self.input_size + self.hidden_size, self.hidden_size).float()
        self.y_h = nn.Linear(self.hidden_size, self.output_size).float()

        self.h_x.apply(self.init_weights)
        self.y_h.apply(self.init_weights)

        self.start_datetime = datetime.datetime.timetuple(datetime.datetime.now())
        self.name = "{}_{}_{}_{}_{}_{}".format(self.start_datetime[1],
                                               self.start_datetime[2],
                                               self.start_datetime[3],
                                               self.start_datetime[4],
                                               self.start_datetime[5],
                                               self.start_datetime[6])
        os.mkdir("models/" + self.name)

    def init_weights(self, m):
        m.weight.data.uniform_(-self.weight_init, self.weight_init)
        m.bias.data.uniform_(-self.weight_init, self.weight_init)

    def init_hidden_state(self):
        hidden = torch.zeros(self.hidden_size, dtype=torch.float32)
        return hidden

    def forward_item(self, x, old_h):
        combined = torch.cat((x, old_h))

        z_h = self.h_x(combined)
        h = self.tanh(z_h)

        z_o = self.y_h(h)
        z_o = z_o.unsqueeze(dim=0)
        o = self.sigmoid(z_o)
        o_prob = torch.nn.functional.softmax(z_o, dim=1)
        return h, z_o, o, o_prob

    def train_item(self, x, old_h, y, torch_optimizer):
        h, z_o, o, o_prob = self.forward_item(x, old_h)
        torch_optimizer.zero_grad()
        index = torch.argmax(y).unsqueeze(dim=0)
        loss = self.criterion(z_o, index)
        loss.backward(retain_graph=True)
        torch_optimizer.step()
        return h, o, o_prob, loss

    def save_model(self):
        file_location = 'models/' + self.name + '/dataset.p'
        outfile = open(file_location, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

        file_location = "models/" + self.name + "/vocab.csv"
        f = open(file_location, 'w')
        for word in self.training_set.vocab_list:
            f.write('{},{}\n'.format(word, self.training_set.vocab_category_dict[word]))
        f.close()

    def load_model(self, name):
        weight_file = "models/" + name + "/weights.csv"
        self.name = name
        weight_file = open(weight_file, 'rb')
        weights_list = pickle.load(weight_file)
        weight_file.close()

        self.h_x = weights_list[0]
        self.y_h = weights_list[1]

from rnn_test import dataset
from rnn_test import network
from rnn_test import analysis


def main():

    num_train_sentences = 100
    num_test_sentences = 10
    num_epochs = 2
    hidden_size = 12
    learning_rate = 0.05
    weight_init = 0.01

    training_dataset = dataset.DataSet(num_train_sentences)
    test_dataset = dataset.DataSet(num_test_sentences)
    my_network = network.SRN()
    my_network.init_model(training_dataset, hidden_size, learning_rate, weight_init)
    analysis.train_model(my_network, training_dataset, test_dataset, num_epochs, learning_rate)
    my_network.save_model()


main()




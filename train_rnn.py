from rnn import dataset
from rnn import network
from rnn import analysis


def main():

    num_train_sentences = 1000
    num_test_sentences = 100
    num_epochs = 10
    hidden_size = 8
    hidden_actf = 'tanh'
    optimizer = 'SGD'
    learning_rate = 0.20
    weight_init = 0.01
    shuffle_sentence = True

    training_dataset = dataset.DataSet(num_train_sentences)
    test_dataset = dataset.DataSet(num_test_sentences)
    my_network = network.SRN()
    my_network.init_model(training_dataset, hidden_size, hidden_actf, learning_rate, weight_init)
    analysis.train_model(my_network, training_dataset, test_dataset, optimizer, num_epochs, learning_rate)


main()



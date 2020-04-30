from rnn_test.networks import srn
from rnn_test.analysis import analysis
from rnn_test.dataset import dataset


def main():

    num_epochs = 2
    hidden_size = 12
    learning_rate = 0.05
    weight_init = 0.01
    shuffle_documents = True
    shuffle_sentences = True
    training_corpus = 'test_2'
    test_corpus = 'test_2'

    training_dataset = dataset.Dataset()
    training_dataset.create_from_corpus(training_corpus)
    training_dataset.generate_one_hot_numpy_vectors()
    training_dataset.convert_to_torch()

    test_dataset = dataset.Dataset()
    test_dataset.create_from_corpus(test_corpus)
    test_dataset.generate_one_hot_numpy_vectors()
    test_dataset.convert_to_torch()

    my_network = srn.SRN()
    my_network.init_model(training_dataset, hidden_size, learning_rate, weight_init)
    analysis.train_model(my_network, training_dataset, test_dataset,
                         num_epochs, learning_rate, shuffle_documents, shuffle_sentences)
    # my_network.save_model()


main()




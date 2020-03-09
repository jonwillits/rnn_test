from rnn_test.networks import srn
from rnn_test.analysis import analysis
from rnn_test.language import simple_animal_world


def main():

    num_epochs = 2
    hidden_size = 12
    learning_rate = 0.05
    weight_init = 0.01
    training_corpus_file = None
    test_corpus_file = None

    training_corpus = simple_animal_world.SimpleAnimalWorld()
    test_corpus = simple_animal_world.SimpleAnimalWorld()

    my_network = srn.SRN()
    my_network.init_model(training_corpus, hidden_size, learning_rate, weight_init)
    analysis.train_model(my_network, training_corpus, test_corpus, num_epochs, learning_rate)
    my_network.save_model()


main()




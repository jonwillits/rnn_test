from rnn_test.language import corpus, simple_animal_world
from rnn_test.dataset import dataset


def main():
    num_documents = 2
    sentences_per_document = 1

    name = 'test_2'

    the_language = simple_animal_world.SimpleAnimalWorld()
    the_language.init_language()

    the_corpus = corpus.Corpus(name)
    the_corpus.generate_corpus(the_language, num_documents, sentences_per_document)
    the_corpus.save_corpus()

    the_dataset = dataset.Dataset()
    the_dataset.save_dataset()


main()


from rnn_test.language import corpus, simple_animal_world
from rnn_test.dataset import dataset


def main():
    num_documents = 5
    sentences_per_document = 10
    use_language_vocab = True

    name = 'test_2'

    the_language = simple_animal_world.SimpleAnimalWorld()
    the_language.init_language()

    the_corpus = corpus.Corpus(name)
    the_corpus.generate_corpus(the_language, num_documents, sentences_per_document)
    the_corpus.save_corpus()

    the_dataset = dataset.Dataset()
    the_dataset.load_corpus(the_corpus, use_language_vocab)
    the_dataset.save_dataset()


main()

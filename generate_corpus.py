from rnn_test.language import corpus, simple_animal_world


def main():
    num_documents = 5
    sentences_per_document = 10

    name = 'test_2'

    the_language = simple_animal_world.SimpleAnimalWorld()
    the_language.init_language()

    the_corpus = corpus.Corpus(name)
    the_corpus.generate_corpus(the_language, num_documents, sentences_per_document)
    the_corpus.save_corpus()


main()

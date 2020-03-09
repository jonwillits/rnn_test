import os
import pickle


class Corpus:

    def __init__(self):
        self.language = None

        self.document_list = None
        self.num_documents = None
        self.num_sentences = None

        self.num_types = 0
        self.num_tokens = 0
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_freq_dict = {}

    def generate_corpus(self, language, num_documents, num_sentences):

        self.language = language

        self.num_documents = num_documents

        self.document_list = []
        self.num_sentences = num_sentences

        for i in range(self.num_documents):
            new_document = Document(i)
            for j in range(self.num_sentences):
                new_sentence = self.language.generate_sentence()

                for token in new_sentence.token_list:
                    if token.label not in self.vocab_index_dict:
                        self.vocab_index_dict[token.label] = self.num_types
                        self.num_types += 1
                        self.vocab_freq_dict[token.label] = 0
                        self.vocab_list.append(token.label)
                    self.vocab_freq_dict[token.label] += 1

                new_document.add_sentence(new_sentence)

            self.document_list.append(new_document)
            self.num_tokens += new_document.num_tokens

        for i in range(self.language.vocab_size):
            word = self.language.vocab_list[i].label
            if word in self.vocab_index_dict:
                print(word, self.vocab_index_dict[word], self.vocab_freq_dict[word])

    def save(self, name):
        try:
            os.mkdir('corpora/' + name)
        except RuntimeError:
            print("Could not generate corpus directory {}. Does it already exist?".format(name))

        f = open('corpora/' + name + '/language_vocab_list.txt', 'w')
        for token in self.language.vocab_list:
            if token.label in self.vocab_freq_dict:
                freq = self.vocab_freq_dict[token.label]
            else:
                freq = 0
            f.write('{} {} {} {}\n'.format(token.index, token.label, token.category, freq))
        f.close()

        f = open('corpora/' + name + '/corpus_vocab_list.txt', 'w')
        for token in self.language.vocab_list:
            if token.label in self.vocab_index_dict:
                index = self.vocab_index_dict[token.label]
                if token.label in self.vocab_freq_dict:
                    freq = self.vocab_freq_dict[token.label]
                else:
                    freq = 0
                f.write('{} {} {} {}\n'.format(index, token.label, token.category, freq))
        f.close()

        f = open('corpora/' + name + '/corpus.txt', 'w')
        for i in range(self.num_documents):
            token_list = []
            for sentence in self.document_list[i].sentence_list:
                for token in sentence.token_list:
                    token_list.append(token.label)
            output_string = ' '.join(token_list) + '\n'
            f.write(output_string)
        f.close()

        file_location = "corpora/" + self.net_name + "/corpus_object.p"
        outfile = open(file_location, 'wb')
        pickle.dump(self)
        outfile.close()

        file_location = "models/" + self.net_name + "/vocab.csv"
        f = open(file_location, 'w')
        for word in self.training_set.vocab_list:
            f.write('{},{}\n'.format(word, self.training_set.vocab_category_dict[word]))
        f.close()

    def load_model(self, model_name):
        weight_file = "models/" + model_name + "/weights.csv"
        self.net_name = model_name
        weight_file = open(weight_file, 'rb')
        weights_list = pickle.load(weight_file)
        weight_file.close()

        self.h_x = weights_list[0]
        self.y_h = weights_list[1]


    def __str__(self):
        output_string = 'Corpus(' + self.language.name + ',' + str(self.num_documents) + ')'
        return output_string


class Document:

    def __init__(self, index):
        self.index = index
        self.num_sentences = 0
        self.num_tokens = 0
        self.sentence_list = []

    def add_sentence(self, sentence):
        self.sentence_list.append(sentence)
        self.num_sentences += 1
        self.num_tokens += sentence.length

    def __str__(self):
        output_string = 'Document(' + self.index + ')\n'
        for sentence in self.sentence_list:
            output_string += sentence
        return output_string

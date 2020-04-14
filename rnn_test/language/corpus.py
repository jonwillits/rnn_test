import os
import shutil
import pickle
import sys


class Corpus:

    def __init__(self, name):
        self.name = name
        self.language = None

        self.document_list = None
        self.num_documents = None
        self.num_sentences = None

        self.vocab_size = 0
        self.num_tokens = 0
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_freq_dict = {}

        if os.path.exists("corpora/" + self.name):
            response = input("Directory {} exists. Replace? (y/n)".format(self.name))
            if response == 'y' or response == 'Y':
                replace = True
            else:
                replace = False
            if replace:
                shutil.rmtree("corpora/" + self.name)
            else:
                sys.exit()

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
                        self.vocab_index_dict[token.label] = self.vocab_size
                        self.vocab_size += 1
                        self.vocab_freq_dict[token.label] = 0
                    self.vocab_freq_dict[token.label] += 1

                new_document.add_sentence(new_sentence)

            self.document_list.append(new_document)
            self.num_tokens += new_document.num_tokens

        counter = 0
        for i in range(self.language.vocab_size):
            word = self.language.vocab_list[i].label
            if word in self.vocab_index_dict:
                self.vocab_list.append(word)
                self.vocab_index_dict[word] = counter
                counter += 1

    def save_corpus(self):

        try:
            os.mkdir('corpora/' + self.name)
        except:
            print("Could not generate corpus directory {}. Does it already exist?".format(self.name))
            sys.exit(2)

        f = open('corpora/' + self.name + '/language_vocab_list.txt', 'w')
        for token in self.language.vocab_list:
            if token.label in self.vocab_freq_dict:
                freq = self.vocab_freq_dict[token.label]
            else:
                freq = 0
            f.write('{} {} {} {}\n'.format(token.index, token.label, token.category, freq))
        f.close()

        f = open('corpora/' + self.name + '/corpus_vocab_list.txt', 'w')
        for token in self.language.vocab_list:
            if token.label in self.vocab_index_dict:
                index = self.vocab_index_dict[token.label]
                if token.label in self.vocab_freq_dict:
                    freq = self.vocab_freq_dict[token.label]
                else:
                    freq = 0
                f.write('{} {} {} {}\n'.format(index, token.label, token.category, freq))
        f.close()

        f = open('corpora/' + self.name + '/corpus.txt', 'w')
        for i in range(self.num_documents):
            token_list = []
            for sentence in self.document_list[i].sentence_list:
                for token in sentence.token_list:
                    token_list.append(token.label)
            output_string = ' '.join(token_list) + '\n'
            f.write(output_string)
        f.close()

        file_location = "corpora/" + self.name + "/corpus_object.p"
        outfile = open(file_location, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def load_corpus(self, name):
        self.name = name
        corpus_file_name = "corpora/" + name + "/corpus_object.p"
        corpus_file = open(corpus_file_name, 'rb')
        corpus_object = pickle.load(corpus_file)
        corpus_file.close()

        self.language = corpus_object.language

        self.document_list = corpus_object.document_list
        self.num_documents = corpus_object.num_documents
        self.num_sentences = corpus_object.num_sentences

        self.vocab_size = corpus_object.vocab_size
        self.num_tokens = corpus_object.num_tokens
        self.vocab_list = corpus_object.vocab_list
        self.vocab_index_dict = corpus_object.vocab_index_dict
        self.vocab_freq_dict = corpus_object.vocab_freq_dict

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

import numpy as np
from rnn_test.language import corpus
import random
import torch
import copy
import pickle


####################################################################################################
####################################################################################################
class Dataset:

    ####################################################################################################
    def __init__(self, name):
        self.name = name
        self.corpus = None
        self.vocab_size = None
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_vector_dict = None
        self.vocab_category_dict = None
        self.num_sequences = None
        self.xy = None
        self.xy_labels = None

    ####################################################################################################
    def create_from_corpus(self, corpus_name, use_lang_vocab=True, numpy_only=False):

        self.corpus = corpus.Corpus(corpus_name)
        self.corpus.load_corpus(corpus_name)

        if use_lang_vocab:
            self.vocab_size = self.corpus.language.vocab_size
            self.vocab_list = []
            self.vocab_index_dict = {}
            self.vocab_category_dict = {}
            for i in range(self.vocab_size):
                token = self.corpus.language.vocab_list[i]
                self.vocab_list.append(token.label)
                self.vocab_index_dict[token.label] = token.index
                self.vocab_category_dict[token.label] = token.category
        else:
            self.vocab_size = self.corpus.vocab_size
            self.vocab_list = self.corpus.vocab_list
            self.vocab_index_dict = self.corpus.vocab_index_dict
            self.vocab_category_dict = {}
            for token in self.corpus.language.vocab_list:
                self.vocab_category_dict[token.label] = token.category

    ####################################################################################################
    def generate_one_hot_numpy_vectors(self):
        self.vocab_vector_dict = {}
        one_hot_numpy_matrix = np.float32(np.eye(self.vocab_size).astype(float))
        for i in range(self.vocab_size):
            self.vocab_vector_dict[self.vocab_list[i]] = one_hot_numpy_matrix[i]

    ####################################################################################################
    def generate_distributed_numpy_vectors(self, size, value):
        for i in range(self.vocab_size):
            self.vocab_vector_dict[self.vocab_list[i]] = np.random.uniform(-value, value, size)

    ####################################################################################################
    def convert_to_torch(self):
        for i in range(self.vocab_size):
            self.vocab_vector_dict[self.vocab_list[i]] = torch.from_numpy(self.vocab_vector_dict[self.vocab_list[i]])

    ####################################################################################################
    def load_dataset(self, name):

        dataset_object_name = "corpora/" + name + "/dataset.p"
        dataset_object_handle = open(dataset_object_name, 'rb')
        dataset_object = pickle.load(dataset_object_handle)
        dataset_object_handle.close()

        self.corpus = dataset_object.corpus
        self.vocab_size = dataset_object.vocab_size
        self.vocab_list = dataset_object.vocab_list
        self.vocab_index_dict = dataset_object.vocab_index_dict
        self.vocab_vector_dict = dataset_object.vocab_vector_dict
        self.num_sequences = dataset_object.num_sequences
        self.xy = dataset_object.x
        self.xy_labels = dataset_object.x_labels

    ####################################################################################################
    def save_dataset(self):
        file_location = 'corpora/' + self.corpus.name + '/dataset.p'
        outfile = open(file_location, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    ####################################################################################################
    def create_xy(self, shuffle_documents, shuffle_sentences):
        self.xy = []
        self.xy_labels = []
        copy_of_vector_docs = copy.deepcopy(self.corpus.document_list)

        if shuffle_documents:
            random.shuffle(copy_of_vector_docs)

        for i in range(len(copy_of_vector_docs)):
            document = copy_of_vector_docs[i]
            sentences = copy.deepcopy(document.sentence_list)
            if shuffle_sentences:
                random.shuffle(sentences)

            for j in range(document.num_sentences):
                sentence = sentences[j]
                for k in range(sentence.length):
                    token = sentence.token_list[k]
                    vector = self.vocab_vector_dict[token.label]
                    index = self.vocab_index_dict[token.label]
                    self.xy_labels.append((index, token.label, token.category))
                    self.xy.append(vector)


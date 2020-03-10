import numpy as np
import random
import torch
import copy
import pickle


class Dataset:

    def __init__(self):
        self.corpus = None
        self.num_types = None
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_vector_dict = None
        self.num_sequences = None
        self.x = None
        self.x_labels = None

    def load_corpus(self, corpus, use_lang_vocab=False):
        self.corpus = corpus
        if use_lang_vocab:
            self.num_types = corpus.language_num_types
            self.vocab_list = corpus.language.vocab_list
            self.vocab_index_dict = corpus.language.vocab_dict
        else:
            self.num_types = corpus.num_types
            self.vocab_list = corpus.vocab_list
            self.vocab_index_dict = corpus.vocab_dict

    def generate_one_hot_numpy_vectors(self):
        one_hot_numpy_matrix = np.float32(np.eye(self.num_types).astype(float))
        for i in range(self.num_types):
            self.vocab_vector_dict[self.vocab_list[i]] = one_hot_numpy_matrix[i]

    def generate_distributed_numpy_vectors(self, size, value):
        for i in range(self.num_types):
            self.vocab_vector_dict[self.vocab_list[i]] = np.random.uniform(-value, value, size)

    def convert_to_torch(self):
        for i in range(self.num_types):
            self.vocab_vector_dict[self.vocab_list[i]] = torch.from_numpy(self.vocab_vector_dict[self.vocab_list[i]])

    def load_dataset(self, name):

        corpus_file_name = "corpora/" + name + "/corpus_object.p"
        corpus_file = open(corpus_file_name, 'rb')
        dataset_object = pickle.load(corpus_file)
        corpus_file.close()

        self.corpus = dataset_object.corpus
        self.num_types = dataset_object.num_types
        self.vocab_list = dataset_object.vocab_list
        self.vocab_index_dict = dataset_object.vocab_index_dict
        self.vocab_vector_dict = dataset_object.vocab_vector_dict
        self.num_sequences = dataset_object.num_sequences
        self.x = None
        self.x_labels = None

    def save_dataset(self):
        file_location = 'corpora/' + self.corpus.name + '/dataset.p'
        outfile = open(file_location, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def create_x(self, shuffle_documents, shuffle_sentences):
        self.x = []
        self.x_labels = []
        copy_of_vector_docs = copy.deepcopy(self.corpus.document_list)

        if shuffle_documents:
            random.shuffle(copy_of_vector_docs)
            for i in range(len(copy_of_vector_docs)):
                doc = copy_of_vector_docs[i]
                if shuffle_sentences:
                    random.shuffle(doc)
                    for j in range(len(doc)):
                        sentence = doc[j]
                        for word in sentence:
                            self.x.append(self.vocab_vector_dict[word.label])
                            self.x_labels.append(word.label)

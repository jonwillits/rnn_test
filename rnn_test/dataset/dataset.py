import numpy as np


class Dataset:

    def __init__(self):
        self.corpus = None
        self.num_types = None
        self.item_list = []
        self.item_index_dict = {}
        self.item_vector_dict = None
        self.num_sentences = None
        self.sequence_list = None

    def load_corpus(self, corpus):
        self.corpus = corpus
        self.num_types = corpus.num_types
        self.item_list = []
        self.item_index_dict = {}

    def generate_one_hot_numpy_vectors(self):
        self.onehot_numpy_ = np.float32(np.eye(self.num_item_types).astype(float))

    def generate_distributed_numpy_vectors(self, length):

    def load_dataset(self):
        pass

    def save_dataset(self):
        pass



    def create_x(self, shuffle_sentences):
        self.x = []
        self.x_words = []
        copy_of_sentences = self.list_of_sentences.copy()
        if shuffle_sentences:
            random.shuffle(copy_of_sentences)

        for sentence in copy_of_sentences:
            for word in sentence:
                word_index = self.vocab_index_dict[word]
                word_vector = self.word_vector_matrix[word_index, :]
                word_tensor = torch.from_numpy(word_vector)
                self.x.append(word_tensor)
                self.x_words.append(word)
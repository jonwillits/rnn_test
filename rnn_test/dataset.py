import random
import numpy as np
import torch


class DataSet:

    def __init__(self, num_sentences):

        self.num_sentences = num_sentences

        self.determiners = ['the', 'a']
        self.herbivores = ['cow', 'deer', 'pig', 'rabbit']
        self.plants = ['carrot', 'lettuce', 'onion', 'grass']
        self.drinks = ['water', 'beer', 'juice', 'milk']
        self.t_verbs = ['chased', 'ate', 'drank', 'watched']
        self.i_verbs = ['slept', 'walked', 'ran', 'swam']
        self.punctuation = ['.']
        self.word_classes = [self.determiners, self.herbivores, self.plants, self.drinks,
                             self.t_verbs, self.i_verbs, self.punctuation]
        self.word_class_string_list = ['determiners', 'herbivores', 'plants', 'drinks', 't_verbs', 'i_verbs',
                                'punctuation']

        self.list_of_sentences = None
        self.num_word_types = 0
        self.num_word_tokens = 0
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_category_dict = {}
        self.word_vector_matrix = None
        self.x = None
        self.x_words = None

        self.init_dataset()
        self.generate_sentences()

    def init_dataset(self):
        for i in range(len(self.word_classes)):
            word_class = self.word_classes[i]
            word_class_string = self.word_class_string_list[i]
            for word in word_class:
                self.vocab_list.append(word)
                self.vocab_category_dict[word] = word_class_string
                self.vocab_index_dict[word] = self.num_word_types
                self.num_word_types += 1
        self.word_vector_matrix = np.float32(np.eye(self.num_word_types).astype(float))

    def generate_sentences(self):
        self.list_of_sentences = []

        for i in range(self.num_sentences):

            # pick an herbivore
            # pick an action
            # if 'chased' or 'watched', pick an herbivore to chase
            # if ate, pick a plant
            # if drank, pick a drink

            agent = random.choice(self.herbivores)
            verb = random.choice(self.t_verbs + self.i_verbs)

            sentence = [random.choice(self.determiners), agent, verb]
            if verb in self.t_verbs:
                if verb == 'ate':
                    patient = random.choice(self.plants)
                elif verb == 'watched' or verb == 'chased':
                    patient = random.choice(self.herbivores)
                elif verb == 'drank':
                    patient = random.choice(self.drinks)
                else:
                    raise RuntimeError
                sentence.append(random.choice(self.determiners))
                sentence.append(patient)

            sentence.append(random.choice(self.punctuation))
            self.list_of_sentences.append(sentence)
            self.num_word_tokens += len(sentence)

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


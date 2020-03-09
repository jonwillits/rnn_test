import random


class Language:

    def __init__(self):
        self.name = "ABCD"
        self.init_dict = {
            'A': ['A1', 'A2', 'A3', 'A4'],
            'B': ['B1', 'B2', 'B3', 'B4'],
            'C': ['C1', 'C2', 'C3', 'C4'],
            'D': ['D1', 'D2', 'D3', 'D4'],
            '.': ['.']
        }

        self.category_list = []
        self.category_dict = {}
        self.num_categories = 0

        self.vocab_list = []
        self.vocab_dict = {}
        self.vocab_size = 0

    def init_language(self):

        for category_label in self.init_dict:
            category_object = Category(category_label, self.num_categories)
            self.num_categories += 1

            for word_label in self.init_dict[category_label]:
                word_object = Token(word_label, category_label, self.vocab_size)
                self.vocab_list.append(word_object)
                self.vocab_dict[word_label] = word_object
                self.vocab_size += 1
                category_object.add_word(word_object)

            self.category_list.append(category_object)
            self.category_dict[category_label] = category_object

        self.category_list = sorted(self.category_list, key=lambda x: x.label, reverse=True)

    def __str__(self):
        output_string = 'Language(' + self.name + ')\n{} Categories:\n'.format(self.num_categories)
        for category in self.category_list:
            output_string += '\t{}\n'.format(category)
        output_string += '\n{} Words:\n'.format(self.vocab_size)
        for token in self.vocab_list:
            output_string += '\t{}\n'.format(token)
        return output_string

    def generate_sentence(self):

        def choose_next(i, category):
            cutoff = random.random()
            if cutoff < 0.75:
                choice = self.category_dict[category].member_list[i]
            else:
                choice = random.choice(self.category_dict[category].member_list)
            return choice

        print("Generating Language for {}".format(self.name))
        sentence = Sentence()
        index = random.randint(0, 3)
        sentence.add_word(self.category_dict['A'].member_list[index])
        sentence.add_word(choose_next(index, 'B'))
        sentence.add_word(choose_next(index, 'C'))
        sentence.add_word(choose_next(index, 'D'))
        sentence.add_word(random.choice(self.category_dict['.'].member_list))
        return sentence

class Category:

    def __init__(self, label, index):
        self.label = label
        self.member_list = []
        self.member_index_dict = {}
        self.size = 0
        self.index = index

    def add_word(self, word):
        self.member_list.append(word)
        self.member_index_dict[word] = self.size
        self.size += 1

    def __str__(self):
        return 'Category(' + str(self.index) + ',' + self.label + ')'


class Token:

    def __init__(self, label, category, index):
        self.label = label
        self.index = index
        self.category = category
        self.freq = 0

    def __str__(self):
        return 'Token(' + str(self.index) + ',' + self.label + ')'


class Sentence:

    def __init__(self):
        self.token_list = []
        self.length = 0

    def add_word(self, word):
        self.token_list.append(word)
        self.length += 1
        word.freq += 1

    def __str__(self):
        output_string = 'Sentence('
        for token in self.token_list:
            output_string += token.label + ' '
        output_string += ')\n'

        return output_string

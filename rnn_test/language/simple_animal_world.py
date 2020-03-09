from rnn_test.language import languages
import random


class SimpleAnimalWorld(languages.Language):

    def __init__(self):
        super().__init__()

        self.name = 'simple_animal_world'

        self.init_dict = {
            'determiners': ['the', 'a'],
            'herbivores': ['cow', 'deer', 'pig', 'rabbit'],

            'i_verbs': ['slept', 'walked', 'ran', 'swam'],

            't_verbs': ['chased', 'ate', 'drank', 'watched'],
            'plants': ['carrot', 'lettuce', 'onion', 'grass'],
            'drinks': ['water', 'beer', 'juice', 'milk'],

            'punctuation': ['.'],
            'unknown': ['UNKNOWN']
        }

    def generate_document(self):
        pass

    def generate_sentence(self):

        sentence = languages.Sentence()

        agent_determiner = random.choice(self.category_dict['determiners'].member_list)
        sentence.add_word(agent_determiner)

        agent = random.choice(self.category_dict['herbivores'].member_list)
        sentence.add_word(agent)

        verb_type = random.choice(['transitive', 'intransitive'])
        if verb_type == 'intransitive':
            verb = random.choice(self.category_dict['i_verbs'].member_list)
            sentence.add_word(verb)
        else:
            verb = random.choice(self.category_dict['t_verbs'].member_list)
            sentence.add_word(verb)

            patient_determiner = random.choice(self.category_dict['determiners'].member_list)
            sentence.add_word(patient_determiner)

            if verb.label == 'ate':
                patient = random.choice(self.category_dict['plants'].member_list)
            elif verb.label == 'watched' or verb.label == 'chased':
                patient = random.choice(self.category_dict['herbivores'].member_list)
            elif verb.label == 'drank':
                patient = random.choice(self.category_dict['drinks'].member_list)
            else:
                raise RuntimeError
            sentence.add_word(patient)

        punctuation = random.choice(self.category_dict['punctuation'].member_list)
        sentence.add_word(punctuation)

        return sentence

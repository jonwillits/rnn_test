
import torch


def train_model(model, training_set, test_set, num_epochs, learning_rate):

    torch_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        training_set.create_x(True)
        h = model.init_hidden_state()
        for i in range(len(training_set.x)-1):
            h, z_o, o, o_prob = model.train_item(training_set.x[i], h, training_set.x[i+1], torch_optimizer)
            if i % 100 == 0:
                print("Finished", e, i)

        evaluate_network(model, test_set)


def evaluate_network(model, test_set):

    test_set.create_x(False)
    h = model.init_hidden_state()
    for i in range(len(test_set.x)-1):
        h, z_o, o, o_prob = model.forward_item(test_set.x[i], h)
        print()
        print("x:", test_set.x_words[i])
        print(test_set.x[i])
        print()
        print("h_x:")
        print(model.h_x.bias)
        print(model.h_x.weight)
        print()
        print("h:", h)
        print()
        print("y_h:")
        print(model.y_h.bias)
        print(model.y_h.weight)
        print()
        print(test_set.x[i+1])
        print(test_set.x_words[i+1])
        print()
        print("o:", o)
        print("o_prob:", o_prob)

        """             CAT1        CAT2        CAT3 ....
            Epoch 1     .04         .04         .04      
            Epoch 2     .05         .04         .04      
            Epoch 3     .08         .04         .04          
        """

        # have some code that evaluated accuracy (in terms of probability of predicting the correct word)
        # broken down by each category

        # for each test item
        #   figure out what output was teh highest, call that the "guess"
        #   compare that to the correct answer, if same, say "correct"
        #   compute % correct for each category in the language
        #       in other words, what percentage of the time, was it "correct", when the right answer
        #       was in each category
        #   result_list should be a list of proportions of length self.word_classes
        #   make a graph of the results

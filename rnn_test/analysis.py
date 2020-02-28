
import torch


def train_model(model, training_set, test_set, optimizer, num_epochs, learning_rate):
    if optimizer == 'Adam':
        torch_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        torch_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        torch_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(num_epochs):
        training_set.create_x(True)
        for i in range(len(training_set.x)-1):
            o, loss = model.train_item(training_set.x[i], training_set.x[i+1], torch_optimizer)

        evaluate_network(model, test_set)


def evaluate_network(model, test_set):

    test_set.create_x(False)
    for i in range(len(test_set.x)-1):
        o, h = model.forward_item(test_set.x[i])
        print()
        print(test_set.x_words[i], test_set.x_words[i+1])
        print(test_set.x[i])
        print(test_set.x[i+1])
        print(model.softmax(o))

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

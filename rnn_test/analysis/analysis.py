
import torch


def train_model(model, training_set, test_set, num_epochs, learning_rate, shuffle_documents, shuffle_sentences, verbose):

    torch_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    mean_loss = evaluate_network(model, test_set, torch_optimizer, verbose)
    print("Training {}-{}-{} {} for {} epochs".format(model.input_size, model.hidden_size, model.output_size,
                                                      model.name, num_epochs))
    print("     Epoch 0 Average Test Loss: {:0.3f}".format(mean_loss))

    for e in range(num_epochs):
        training_set.create_xy(shuffle_documents, shuffle_sentences)

        h = model.init_hidden_state()
        for i in range(len(training_set.xy)-1):
            h, o, o_prob, loss = model.train_item(training_set.xy[i], h, training_set.xy[i+1], torch_optimizer)
        mean_loss = evaluate_network(model, test_set, torch_optimizer, verbose)
        print("     Epoch {} Average Test Loss: {:0.3f}".format(e+1, mean_loss))
        model.save_performance([e, mean_loss.detach().numpy()])


def evaluate_network(model, test_set, torch_optimizer, verbose):
    test_set.create_xy(False, False)

    h = model.init_hidden_state()
    summed_loss = 0
    num_items = len(test_set.xy)-1

    for i in range(num_items):
        h, z_o, o, o_prob, loss = model.test_item(test_set.xy[i], h, test_set.xy[i+1], torch_optimizer)

        o_probs = o_prob.detach().numpy()[0]
        o_probs_max = o_probs.max()
        summed_loss += loss
        if verbose:
            if i % 10 == 0:
                print("\n         x          y          loss    y_act   max_act   category_act")
            x_index = test_set.xy_labels[i][0]
            y_index = test_set.xy_labels[i+1][0]
            x_label = test_set.xy_labels[i][1]
            y_label = test_set.xy_labels[i+1][1]
            x_category = test_set.xy_labels[i][2]
            y_category = test_set.xy_labels[i+1][2]
            print("         {:10} {:10} {:0.3f}   {:0.3f}   {:0.3f}".format(x_label, y_label, loss, o_probs[y_index], o_probs_max))

    mean_loss = summed_loss / num_items
    return mean_loss
from rnn_test import display
from rnn_test import network


def main():

    model_name = '3_6_14_58_6_4'

    the_network = network.SRN()
    the_network.load_model(model_name)
    print(the_network.y_h.bias)
    the_display = display.Display(the_network)
    the_display.root.mainloop()


main()

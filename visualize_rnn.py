from rnn_test.display import display
from rnn_test.networks import srn


def main():

    model_name = '3_6_14_58_6_4'

    the_network = srn.SRN()
    the_network.load_model(model_name)
    the_display = display.Display(the_network)
    the_display.root.mainloop()


main()

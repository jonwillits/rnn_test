import tkinter as tk
from tkinter import ttk
import sys
from tkinter import *

button_frame_height = 40
content_frame_width = 800
content_frame_height = 500


class Display(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.height = 900
        self.width = 1200
        self.entry_height = 50
        self.button_height = 20

        container = tk.Frame(self, height=self.height, width=self.width, bd=0, padx=0, pady=0)
        container.pack(side="top", fill="both", expand=True)

        self.main_frames = {}
        for F in (subframe1, subframe2, subframe3):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.main_frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("subframe1")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.main_frames[page_name]
        frame.tkraise()

    def create_buttom(self, page, controller):
        button_frame = tk.Frame(page, height=button_frame_height, width=content_frame_width)
        button_frame.pack(side=TOP)

        load_button = tk.Button(button_frame, text="Load network")
        load_button.pack(side=LEFT, padx=8)
        save_button = tk.Button(button_frame, text="Save network")
        save_button.pack(side=LEFT, padx=8)

        sub1_button = tk.Button(button_frame, text="Go to the subframe1",
                                command=lambda: controller.show_frame("subframe1"))
        sub1_button.pack(side=LEFT, padx=8)
        sub2_button = tk.Button(button_frame, text="Go to the subframe2",
                                command=lambda: controller.show_frame("subframe2"))
        sub2_button.pack(side=LEFT, padx=8)
        sub3_button = tk.Button(button_frame, text="Go to the subframe3",
                                command=lambda: controller.show_frame("subframe3"))
        sub3_button.pack(side=LEFT, padx=8)
        quit_button = ttk.Button(button_frame, text="Quit", width=8, command=sys.exit)
        quit_button.pack(side=tk.LEFT, padx=8)

    def load_network(self):
        pass

    def save_network(self):
        pass

    def train_network(self):
        pass


class subframe1(tk.Frame):
    # 		content_frame1
    # 			- what are the activations in H and Y for the current X
    # 		user_input_frame1
    # 			- has a text input field where you can type a word or string of words
    # 			- ‘go’ button, that calls update for this subframe only, which redraws this subframe only
    # 		update_screen()
    # 			erases everything in main_frame
    # 			draws all subframe1 objects

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="subframe1")
        label.pack(side="top", fill="x", pady=5)

        content_frame1 = tk.Frame(self, height=content_frame_height, width=content_frame_width, bd=0, padx=0, pady=0)
        content_frame1.pack(side=BOTTOM)
        graph_canvas = tk.Canvas(content_frame1, height=content_frame_height, width=content_frame_width,
                                 bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        graph_canvas.pack()

        controller.create_buttom(self, controller)

        user_input_frame1 = tk.Frame(self, height=button_frame_height, width=content_frame_width)
        user_input_frame1.pack(side=TOP)
        input_box = tk.Entry(user_input_frame1, width=40, bd=0)
        input_box.place(relx=0.3)
        input_box.focus_set()

        text_label = Label(user_input_frame1, text="Input: ")
        text_label.place(relx=0.2)
        show_activation_button = tk.Button(user_input_frame1, text="show_activation",
                                           command=lambda: self.retrieve_input(input_box.get()))
        show_activation_button.place(relx=0.8)

    def retrieve_input(self, input_string):
        if not input_string:
            print("Please enter a string")
        else:
            print(input_string)


class subframe2(tk.Frame):
    # 	class subframe2
    # 		content_frame2
    # 			- plot of error of the network learning over time
    # 			- maybe some other plots too, like accuracy on specific words
    # 		user_input_frame1
    # 			- train buttons that train for a specified number of epochs, plotting change in error every n epochs
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="subframe2")
        label.pack(side="top", fill="x", pady=5)

        content_frame2 = tk.Frame(self, height=content_frame_height, width=content_frame_width, bd=0, padx=0, pady=0)
        content_frame2.pack(side=BOTTOM)

        graph_canvas = tk.Canvas(content_frame2, height=content_frame_height, width=content_frame_width,
                                 bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        graph_canvas.pack()

        controller.create_buttom(self, controller)

        button_frame2 = tk.Frame(self, height=button_frame_height, width=content_frame_width)
        button_frame2.pack(side=TOP)

        train_button = tk.Button(button_frame2, text="Train")
        train_button.pack(side=LEFT, padx=8)


class subframe3(tk.Frame):
    # 	class subframe3
    # 		content_frame3
    # 			- some sort of analyses of the hidden states
    # 				dog = cat, shoe = sock
    # 		button_frame3
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="subframe3")
        label.pack(side="top", fill="x", pady=5)

        content_frame3 = tk.Frame(self, height=content_frame_height, width=content_frame_width, bd=0, padx=0, pady=0)
        content_frame3.pack(side=BOTTOM)

        graph_canvas = tk.Canvas(content_frame3, height=content_frame_height, width=content_frame_width,
                                 bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        graph_canvas.pack()

        controller.create_buttom(self, controller)


if __name__ == "__main__":
    display = Display()
    display.mainloop()

# class display
# 	main_frame
# 	subframe1
# 	subframe2
# 	subframe3
# 	button_frame
# class main_frame
# 	class subframe1
# 		content_frame1
# 			- what are the activations in H and Y for the current X
# 		user_input_frame1
# 			- has a text input field where you can type a word or string of words
# 			- ‘go’ button, that calls update for this subframe only, which redraws this subframe only
# 		update_screen()
# 			erases everything in main_frame
# 			draws all subframe1 objects
# 	class subframe2
# 		content_frame2
# 			- plot of error of the network learning over time
# 			- maybe some other plots too, like accuracy on specific words
# 		user_input_frame1
# 			- train buttons that train for a specified number of epochs, plotting change in error every n epochs
# 	class subframe3
# 		content_frame3
# 			- some sort of analyses of the hidden states
# 				dog = cat, shoe = sock
# 		button_frame3
# class button_frame
# 	load network button
# 	save network button
# 	sf1 button (calls subframe1.update_screen())
# 	sf2 button (calls subframe2.update_screen())
# 	sf3 button (calls subframe3.update_screen())
# 	quit button
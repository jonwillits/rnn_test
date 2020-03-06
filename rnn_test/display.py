import tkinter as tk
from tkinter import ttk
import sys


class Display:

    def __init__(self, the_network):
        # todo make sure the properties of the dataset (rows, columns, maybe other things are same as network's

        self.the_network = the_network

        self.height = 900
        self.width = 1200

        self.root = tk.Tk()
        self.root.title("Visualize SRN")

        self.entry_frame = tk.Frame(self.root, height=200, width=self.width, bd=0, padx=0, pady=0)
        self.graph_frame = tk.Frame(self.root, height=self.height-220, width=self.width, bd=0, padx=0, pady=0)
        self.button_frame = tk.Frame(self.root, height=20, bg="white", width=self.width, bd=0, padx=0, pady=0)

        self.entry_frame.pack()
        self.graph_frame.pack()
        self.button_frame.pack()

        self.graph_canvas = tk.Canvas(self.graph_frame, height=self.height-220, width=self.width,
                                      bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        self.graph_canvas.pack()

        ttk.Style().configure("TButton", padding=0, relief="flat", background="#EEEEEE", foreground='black')
        self.quit_button = ttk.Button(self.button_frame, text="Quit", width=8, command=sys.exit)
        self.quit_button.pack(side=tk.LEFT, padx=4)


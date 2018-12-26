# Import Libraries
import numpy as np
import random

# Class to define input layer
class input_layer():
    def __init__(self, no_input_nodes = 0):
        self.length = no_input_nodes
        self.input_layer_weights = {}
        self.input_out = {}
        for i in range(no_input_nodes):
            self.input_layer_weights[i] = np.array([])
            self.input_out[i] = 0

# Class to define hidden layer
class hidden_layer():
    def __init__(self, no_hidden_nodes = 0):
        self.length = no_hidden_nodes
        self.hidden_layer_weights = {}
        self.hidden_out = {}
        for i in range(no_hidden_nodes):
            self.hidden_layer_weights[i] = np.array([])
            self.hidden_out[i] = 0

# Class to define output layer
class output_layer():
    def __init__(self, no_output_nodes = 0):
        self.length = no_output_nodes
        self.output_out = {}
        for i in range(no_output_nodes):
            self.output_out[i] = 0

#Weight Initialization for different layers
class weight_init():
    def __init__(self, inp_obj, hid_obj1, hid_obj2, out_obj, layer = None):
        if layer is 'input':
            self.input_init(inp_obj, hid_obj1)
        elif layer is 'hidden':
            self.hidden_init(hid_obj1, hid_obj2)
        elif layer is 'output':
            self.out_init(hid_obj2, out_obj)
    def input_init(self, inp_obj, hid_obj1):
        inp_len = inp_obj.length
        hid_len = hid_obj1.length
        for i in range(inp_len):
            x = np.array([])
            for j in range(hid_len):
                x = np.append(x, random.uniform(0.05, .1))
            inp_obj.input_layer_weights[i] = x
    def hidden_init(self, hid_obj1, hid_obj2):
        hid1_len = hid_obj1.length
        hid2_len = hid_obj2.length
        for i in range(hid1_len):
            x = np.array([])
            for j in range(hid2_len):
                x = np.append(x, random.uniform(0.05, .1))
            hid_obj1.hidden_layer_weights[i] = x
    def out_init(self, hid_obj2, out_obj):
        hid2_len = hid_obj2.length
        out_len = out_obj.length
        for i in range(hid2_len):
            x = np.array([])
            for j in range(out_len):
                x = np.append(x, random.uniform(0.05, .1))
            hid_obj2.hidden_layer_weights[i] = x

# Designing the ANN
class ANN():
    def __init__(self):
        self.objects = np.array([])
        self.inp_flag = False
        self.out_flag = False
        self.no_of_hidden_layers = 0
    def add_input_layer(self, no_of_neurons = 0):
        if not self.inp_flag:
            self.inp_layer = input_layer(no_input_nodes = no_of_neurons)
            self.objects = np.append(self.objects, self.inp_layer)
            self.inp_flag = True
        else:
            print('Input Layer already added!')
    def add_hidden_layer(self, no_of_neurons = 0):
        self.hid_layer = hidden_layer(no_hidden_nodes = no_of_neurons)
        self.objects = np.append(self.objects, self.hid_layer)
        self.no_of_hidden_layers += 1
    def add_output_layer(self, no_of_neurons = 0):
        if not self.out_flag:
            self.out_layer = output_layer(no_output_nodes = no_of_neurons)
            self.objects = np.append(self.objects, self.out_layer)
            self.out_flag = True
        else:
            print('Output Layer already added!')
    def compile_(self):
        self.weight = weight_init(inp_obj = self.objects[0], hid_obj1 = self.objects[1], hid_obj2 = None, out_obj = None, layer = 'input')
        for i in range(self.no_of_hidden_layers-1):
            self.weight = weight_init(inp_obj = None, hid_obj1 = self.objects[i+1], hid_obj2 = self.objects[i+2], out_obj = None, layer = 'hidden')
        self.weight = weight_init(inp_obj = None, hid_obj1 = None, hid_obj2 = self.objects[-2], out_obj = self.objects[-1], layer = 'output')
    def show_weights(self):
        print('Input Layer Weights:')
        print(self.objects[0].input_layer_weights)
        for i in range(self.no_of_hidden_layers):
            print('Hidden Layer ', i+1,' weights:')
            print(self.objects[i+1].hidden_layer_weights)

# Building an ANN
a = ANN()
a.add_input_layer(no_of_neurons=4096)
a.add_hidden_layer(no_of_neurons=16)
a.add_hidden_layer(no_of_neurons=8)
a.add_hidden_layer(no_of_neurons=32)
a.add_hidden_layer(no_of_neurons=16)
a.add_hidden_layer(no_of_neurons=8)
a.add_hidden_layer(no_of_neurons=32)
a.add_hidden_layer(no_of_neurons=16)
a.add_hidden_layer(no_of_neurons=8)
a.add_hidden_layer(no_of_neurons=32)
a.add_hidden_layer(no_of_neurons=16)
a.add_hidden_layer(no_of_neurons=8)
a.add_hidden_layer(no_of_neurons=32)
a.add_hidden_layer(no_of_neurons=64)
a.add_hidden_layer(no_of_neurons=128)
a.add_output_layer(no_of_neurons=2)
a.compile_()
a.show_weights()

# Import Libraries
import numpy as np
import random
#random.seed(10)

# Class to define input layer
class input_layer():
    def __init__(self, no_input_nodes = 0):
        self.length = no_input_nodes
        self.input_layer_weights = {}
        self.input_out = {}
        self.error = {}
        for i in range(no_input_nodes):
            self.input_layer_weights[i] = np.array([])
            self.input_out[i] = 0
            self.error[i] = 0

# Class to define hidden layer
class hidden_layer():
    def __init__(self, no_hidden_nodes = 0):
        self.length = no_hidden_nodes
        self.hidden_layer_weights = {}
        self.hidden_out = {}
        self.error = {}
        for i in range(no_hidden_nodes):
            self.hidden_layer_weights[i] = np.array([])
            self.hidden_out[i] = 0
            self.error[i] = 0

# Class to define output layer
class output_layer():
    def __init__(self, no_output_nodes = 0):
        self.length = no_output_nodes
        self.output_out = {}
        self.error = {}
        for i in range(no_output_nodes):
            self.output_out[i] = 0
            self.error[i] = 0

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
        self.activation_function = None
        self.pred = []
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
    def compile_(self, X, y, batch_size = 1, epoch_size = 1, learning_rate = 0.3, activation_function = 'sigmoid'):
        # Weight Initialization
        X = X.reset_index(drop = True)
        self.activation_function = activation_function
        weight_init(inp_obj = self.objects[0], hid_obj1 = self.objects[1], hid_obj2 = None, out_obj = None, layer = 'input')
        for i in range(self.no_of_hidden_layers-1):
            weight_init(inp_obj = None, hid_obj1 = self.objects[i+1], hid_obj2 = self.objects[i+2], out_obj = None, layer = 'hidden')
        weight_init(inp_obj = None, hid_obj1 = None, hid_obj2 = self.objects[-2], out_obj = self.objects[-1], layer = 'output')
        for i in range(epoch_size):
            self.pred = []
            print("Executing epoch ",i+1)
            data = X.copy()
            y_batch = np.array(y.copy())
            while(len(data) > 0):
                try:
                    cur = data.iloc[0:batch_size, :].values
                    cur_y = y_batch[:batch_size]
                    data.drop([i for i in range(batch_size)], axis = 0, inplace = True)
                    y_batch = y_batch[batch_size:]
                    data = data.reset_index(drop = True)
                except:
                    cur = data.iloc[:, :].values
                    cur_y = y_batch[:batch_size]
                    data.drop([i for i in range(len(data))], axis = 0, inplace = True)
                    y_batch = y_batch[batch_size:]
                    data.reset_index(drop = True)
                self.forward_propagate(cur, activation_function)
                self.calculate_error(cur_y, self.objects[-1], learning_rate)
            self.training_accuracy(y)
    def training_accuracy(self, y):
        count = 0
        for i in range(len(self.pred)):
            if self.pred[i] == y[i]:
                count += 1
        print("{:.2f}".format((count / len(self.pred)) * 100))
    def all_error_zero(self):
        for i in range(len(self.objects)):
            for j in self.objects[i].error:
                self.objects[i].error[j] = 0
    def calculate_error(self, y, layer, learning_rate):
        for j in y:
            for i in range(layer.length):
                if i == j:
                    layer.error[i] += layer.output_out[i] * (1 - layer.output_out[i]) ** 2
                else:
                    layer.error[i] += (layer.output_out[i] ** 2) * (-1) * (1 - layer.output_out[i])
        for i in range(len(self.objects) - 2, 0, -1):
            prev_layer = layer
            layer = self.objects[i]
            for j in range(layer.length):
                sigma_weight_error = self.calculate_linear_combination(layer.hidden_layer_weights[j], prev_layer.error)
                layer.error[j] = layer.hidden_out[j] * (1 - layer.hidden_out[j]) * sigma_weight_error
                for k in range(len(layer.hidden_layer_weights[j])):
                    layer.hidden_layer_weights[j][k] += learning_rate * prev_layer.error[k] * layer.hidden_out[j]          
        layer = self.objects[0]
        prev_layer = self.objects[1]
        for j in range(layer.length):
            for k in range(len(layer.input_layer_weights[j])):
                layer.input_layer_weights[j][k] += learning_rate * prev_layer.error[k] * layer.input_out[j]
        self.all_error_zero()
    
    def forward_propagate(self, X, activation_function):
        for instance in X:
            for i in range(self.objects[0].length):
                self.objects[0].input_out[i] = instance[i]
            out = self.objects[0].input_out
            for i in range(self.objects[1].length):
                self.objects[1].hidden_out[i] = self.activate(val = self.calculate_linear_combination([self.objects[0].input_layer_weights[j][i] for j in range(self.objects[0].length)], out))
            out = self.objects[1].hidden_out
            for i in range(1, len(self.objects) - 2):
                for j in range(self.objects[i+1].length):
                    self.objects[i+1].hidden_out[j] = self.activate(self.calculate_linear_combination([self.objects[i].hidden_layer_weights[k][j] for k in range(self.objects[i].length)], out))
                out = self.objects[i+1].hidden_out            
            for i in range(self.objects[-1].length):
                self.objects[-1].output_out[i] = self.activate(self.calculate_linear_combination([self.objects[-2].hidden_layer_weights[j][i] for j in range(self.objects[-2].length)], out))
            self.pred.append(np.argmax(np.array(list(self.objects[-1].output_out.values()))))
                    
    def activate(self, val, activation_function = 'sigmoid'):
        if activation_function == 'sigmoid':
            return (1 / (1 + np.exp(-1 * val)))
    def calculate_linear_combination(self, weights, inputs):
        s = 0
        for i in inputs.keys():
            s += inputs[i] * weights[i]
        return s
    def show_weights(self):
        print('Input Layer Weights:')
        print(self.objects[0].input_layer_weights)
        for i in range(self.no_of_hidden_layers):
            print('Hidden Layer ', i+1,' weights:')
            print(self.objects[i+1].hidden_layer_weights)
    def predict(self, X):
        X = X.reset_index(drop = True)
        X = X.iloc[:, :].values
        print(len(X))
        pred = []
        for instance in X:
            #print(X[i:i+1, :])
            #self.forward_propagate(X[i:i+1, :], self.activation_function)
            #pred.append(self.objects[-1].output_out)
            print(instance)
            for i in range(self.objects[0].length):
                self.objects[0].input_out[i] = instance[i]
            out = self.objects[0].input_out
            for i in range(self.objects[1].length):
                self.objects[1].hidden_out[i] = self.activate(val = self.calculate_linear_combination([self.objects[0].input_layer_weights[j][i] for j in range(self.objects[0].length)], out))
                print('Hello')
            out = self.objects[1].hidden_out
            for i in range(1, len(self.objects) - 2):
                for j in range(self.objects[i+1].length):
                    self.objects[i+1].hidden_out[j] = self.activate(self.calculate_linear_combination([self.objects[i].hidden_layer_weights[k][j] for k in range(self.objects[i].length)], out))
                out = self.objects[i+1].hidden_out            
            for i in range(self.objects[-1].length):
                self.objects[-1].output_out[i] = self.activate(self.calculate_linear_combination([self.objects[-2].hidden_layer_weights[j][i] for j in range(self.objects[-2].length)], out))
            
# Building an ANN
import pandas as pd

"""# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
X = pd.DataFrame(X)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
"""

dataset = pd.read_csv('wine.csv')
dataset.quality = dataset.quality.apply(lambda x: x-3)
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
a = ANN()
a.add_input_layer(no_of_neurons=11)
a.add_hidden_layer(no_of_neurons=2)
#a.add_hidden_layer(no_of_neurons=3)
a.add_output_layer(no_of_neurons=5)
a.compile_(X_train, y_train, batch_size = 1, epoch_size = 10, learning_rate = 0.001, activation_function = 'sigmoid')
#pred = a.predict(X_test)


#a.show_weights()
#X.drop([i for i in range(10)], inplace = True)
#X = X.reset_index(drop = True)

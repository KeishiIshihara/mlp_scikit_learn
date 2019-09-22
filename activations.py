#=============================================
# This class defines list of layers
# where activation function are different
#=============================================
from __future__ import print_function
from sknn.mlp import Layer

class ActivationLayers(object):
    def __init__(self, units=10):
        self.units = units
        self.activations_list = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

    def set_activation(self, activation='Rectifier'):
        if not activation in self.activations_list:
            print('Selected activation is not valid.')
            return False
        print('* Activation func is now set as: ',activation)
        # fix with 4 hidden layers 10 neurons
        self.layers = [Layer(activation, units=self.units), 
                       Layer(activation, units=self.units),
                       Layer("Softmax")]

    def get_layers(self):
        return self.layers

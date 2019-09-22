#=============================================
# This class defines list of layers
# those dropout rate are different
#=============================================

from sknn.mlp import Layer

class DropoutLayers(object):
    def __init__(self, units=10):
        self.units = units

    def set_dropouts(self, dropouts):
        # fix with 4 hidden layers 10 neurons
        self.layers = [Layer('Rectifier', units=self.units, dropout=dropouts), 
                       Layer('Rectifier', units=self.units, dropout=dropouts),
                       Layer('Rectifier', units=self.units, dropout=dropouts),
                       Layer('Rectifier', units=self.units, dropout=dropouts),
                       Layer("Softmax")]

    def get_layers(self):
        return self.layers

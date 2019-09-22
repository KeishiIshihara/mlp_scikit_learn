#=============================================
# This class defines list of layers
# which consist of defferent number of layers
#
#  How does the performance improve when the number 
#  of layers increase gradually from 1 to 10
#=============================================

from sknn.mlp import Layer

class myLayers(object):
    def __init__(self, units=10):
        # trial 1 with 1 hidden layer
        layers_1 = [Layer('Rectifier', units=units),
                    Layer("Softmax")]

        # trial 2 with 2 hidden layer
        layers_2 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        # trial 3 and so on..
        layers_3 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_4 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_5 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_6 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]


        layers_7 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_8 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_9 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        layers_10 = [Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer('Rectifier', units=units),
                    Layer("Softmax")]

        self.layers_list = [layers_1,layers_2,layers_3,layers_4,layers_5,layers_6,layers_7,layers_8,layers_9,layers_10]
    
    def get_layers_list(self):
        return self.layers_list
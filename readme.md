# MLP classification with Sklearn

This repository is made for myself to better understand scikit-learn, scikit-neuralnetwork and multi-layer perceptron.

### Requirements
- scikit-learn: Version 0.19.0
- scikit-neuralnetwork: Version 0.7

## train.py
- module: scikit-learm
- sklearn.neural_network.MLPClassifier

## trial_for_neurons.py
- module: scikit-neuralnetwork
    - pros: dropout is avairable for each layer
- sknn.mlp.Classifier
- executes one hidden layer with (5,10,..,30) nodes (10 trials for each nodes)
- outputs its result to graph and csv 
- `trial_for_layers.py`:
    - train with 1 to 10 layers with fixed nodes (10 nodes) (10 trials for each layer)
    - outputs result fig and csv

- `trial_for_dropouts.py`:
    - train with 0% to 40% dropout rate for each layer where are 4 hidden layers 10 nodes (10 trials for each rate)

- `trial_for_activations.py`:
    - train with 3 activation functions: Rectifier(relu), Sigmoid, Tanh (10 trials each) and evaluate them

---
#### Reference
- [scikit-neuralnetwork](https://scikit-neuralnetwork.readthedocs.io/en/latest/module_mlp.html#layer-specifications)
- [matplotlib](https://matplotlib.org/contents.html)
#### Troubleshooting
- sklearn.cross_validation
    - [ImportError: No module named 'sklearn.cross_validation'の対処](https://www.haya-programming.com/entry/2018/12/04/052713)
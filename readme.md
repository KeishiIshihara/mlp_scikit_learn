# MLP classification with Sklearn

This repository is made for myself to better understand scikit-learn, scikit-neuralnetwork and multi-layer perceptron.

## train.py
- module: scikit-learm
- sklearn.neural_network.MLPClassifier

## train_sknn.py
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


---
#### Reference
- [scikit-neuralnetwork](https://scikit-neuralnetwork.readthedocs.io/en/latest/module_mlp.html#layer-specifications)

#### Troubleshooting
- sklearn.cross_validation
    - [ImportError: No module named 'sklearn.cross_validation'の対処](https://www.haya-programming.com/entry/2018/12/04/052713)
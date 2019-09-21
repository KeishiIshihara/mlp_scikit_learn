# MLP classification with Sklearn

This repository is made for myself to better understand scikit-learn, scikit-neuralnetwork and multi-layer perceptron.

## train.py
- sklearn.neural_network.MLPClassifier

## train_sknn.py
- sknn.mlp.Classifier
- executes one hidden layer with (5,10,..,30) nodes
- outputs its result to graph and csv 

---
#### Reference
- [scikit-neuralnetwork](https://scikit-neuralnetwork.readthedocs.io/en/latest/module_mlp.html#layer-specifications)

#### Troubleshooting
- sklearn.cross_validation
    - [ImportError: No module named 'sklearn.cross_validation'の対処](https://www.haya-programming.com/entry/2018/12/04/052713)
# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from gen_data import load_data



X_train, X_test, y_train, y_test = load_data()
clf = MLPClassifier(solver="sgd",random_state=0,max_iter=10000)
clf.fit(X_train, y_train)
print (clf.score(X_test, y_test))
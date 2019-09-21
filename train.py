# coding: utf-8
#======================================
#  Breast cancer tumor classification
#  with sklearn MLPClassifier module
#   (c) Keishi Ishihara
#======================================
from __future__ import print_function

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data


X_train, X_test, y_train, y_test = load_data()

clf = MLPClassifier(hidden_layer_sizes=(30),
                    activation='relu',
                    alpha=1e-5,
                    learning_rate_init=0.001,
                    learning_rate='constant',
                    solver='sgd',
                    random_state=0,
                    verbose=True,
                    tol=1e-4,
                    max_iter=10000)

clf.fit(X_train, y_train)

print ('Training set score: {}'.format(clf.score(X_train, y_train)))
print ('Test set score: {}'.format(clf.score(X_test, y_test)))

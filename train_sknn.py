# coding: utf-8
#======================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module
#
#   (c) Keishi Ishihara
#======================================
from __future__ import print_function

from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer

X_train, X_test, y_train, y_test = load_data()

clf = Classifier(
    layers=[
        Layer('Rectifier', units=20, dropout=0.05),
        Layer('Rectifier', units=20, dropout=0.05),
        Layer('Rectifier', units=20, dropout=0.05),
        Layer('Rectifier', units=20, dropout=0.05),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=100)

clf.fit(X_train, y_train)

print ('Training set score: {}'.format(clf.score(X_train, y_train)))
print ('Test set score: {}'.format(clf.score(X_test, y_test)))

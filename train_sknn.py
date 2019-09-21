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

X_train, X_test, y_train, y_test = load_data(test_size=0.1)

units = [5,10,15,20,25,30]
scores_train = []
scores_test = []

for i, unit in enumerate(units):
    clf = Classifier(
        layers=[
            Layer('Rectifier', units=unit),
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=100)

    clf.fit(X_train, y_train)

    print ('====================================')
    scores_train.append(clf.score(X_train, y_train))
    scores_test.append(clf.score(X_test, y_test))

    print ('num of units >> {}'.format(unit))
    print ('  - Training set score: {}'.format(scores_train[i]))
    print ('  - Test set score: {}'.format(scores_test[i]))


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8)) # Initialize Figure
ax = fig.add_subplot(1,1,1) # make axes
line1, = ax.plot(units, scores_train, label="score for training") # Axes.linesにLine2Dを追加+その他の設定
line2, = ax.plot(units, scores_test, label="score for test") # Axes.linesにLine2Dを追加+その他の設定
ax.legend()
ax.set_title('score')
ax.set_xlabel('units')
ax.set_ylabel('accuracy')
fig.savefig('training_result.png')
plt.close()

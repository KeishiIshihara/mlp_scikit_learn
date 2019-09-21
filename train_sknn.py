# coding: utf-8
#======================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module
#
#   (c) Keishi Ishihara
#======================================
from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer


# units = [5,10,15,20,25,30]
units = [5,10,15]
num_of_experiment = 2 

def add_elementwise(list1, list2):
    print('list1: ',list1)
    print('list2: ',list2)

    if len(list1) != len(list2):
        return list2

    sum = np.array(list1) + np.array(list2)
    return sum.tolist()

# load data
X_train, X_test, y_train, y_test = load_data(test_size=0.1)

result_train = []
result_test = []
for n in range(num_of_experiment):
    scores_train = []
    scores_test = []
    print('=========== experiment: {} ============'.format(n+1))
    for i, unit in enumerate(units):
        clf = Classifier(
            layers=[
                Layer('Rectifier', units=unit),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=10)

        clf.fit(X_train, y_train)

        print ('====================================')
        scores_train.append(clf.score(X_train, y_train))
        scores_test.append(clf.score(X_test, y_test))
        print ('num of units >> {}'.format(unit))
        print ('  - Training set score: {}'.format(scores_train[i]))
        print ('  - Test set score: {}'.format(scores_test[i]))

    result_train = add_elementwise(result_train, scores_train)
    result_test = add_elementwise(result_test, scores_test)
    print('result_train: ',result_train)

result_train = np.array(result_train) / num_of_experiment
result_test = np.array(result_test) / num_of_experiment

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8)) # Initialize Figure
ax = fig.add_subplot(1,1,1) # make axes
line1, = ax.plot(units, result_train, label="score for training") # plot training
line2, = ax.plot(units, result_test, label="score for test") # plot test
ax.legend() # add legend
ax.set_title('score')
ax.set_xlabel('units')
ax.set_ylabel('accuracy')
fig.savefig('training_result.png')
print('line1:',line1)
print('line2:',line2)
plt.close()

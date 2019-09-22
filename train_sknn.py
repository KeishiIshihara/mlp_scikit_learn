# coding: utf-8
#=============================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module  (python2)
#
#   Train with one layer with nodes (5,10,..,30)
#
#   (c) Keishi Ishihara
#=============================================
from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer

# configurations # default
num_of_trials = 10  #10
test_size = 0.3  #0.2
iteration = 25  #25
learning_rate = 0.001  #0.001
units = [5,10,15,20,25,30]

# load data
X_train, X_test, y_train, y_test = load_data(test_size=test_size)
# arrays for storing results
scores_train = [[] for i in range(num_of_trials)]
scores_test = [[] for i in range(num_of_trials)]

for n in range(num_of_trials):
    print('============== Trial: {} ==============='.format(n+1))
    for i, unit in enumerate(units):
        clf = Classifier(
            layers=[
                Layer('Rectifier', units=unit),
                Layer("Softmax")],
            learning_rate=learning_rate,
            n_iter=iteration)

        clf.fit(X_train, y_train)

        print ('====================================')
        scores_train[n].append(clf.score(X_train, y_train))
        scores_test[n].append(clf.score(X_test, y_test))
        print ('num of units >> {}'.format(unit))
        print ('  - Training set score: {}'.format(scores_train[n][i]))
        print ('  - Test set score: {}'.format(scores_test[n][i]))


scores_train = np.array(scores_train)
scores_test = np.array(scores_test)
print('')
print('train:', scores_train)
print('test:', scores_test)

average_train = np.sum(scores_train, axis=0) / float(num_of_trials)
average_test = np.sum(scores_test, axis=0) / float(num_of_trials)


import matplotlib.pyplot as plt
fig = plt.figure() # Initialize Figure
ax = fig.add_subplot(1,1,1) # make axes

for i in range(num_of_trials): # draw all result with light color
    ax.plot(units, scores_train[i], color='blue', linestyle='--', linewidth=0.2) # linestyle is not working
    ax.plot(units, scores_test[i],color='orange', linestyle='--', linewidth=0.25)

line1, = ax.plot(units, average_train, color='blue', label="Average score of training") # plot training
line2, = ax.plot(units, average_test, color='orange', label="Average score of test") # plot test
ax.legend(loc='best') # add legend
ax.set_xticks(range(units[0],units[-1]+5,5))
ax.set_xticklabels(units)
ax.grid(which='both')
ax.set_title('Score (train:test={}:{}, Trials={})'.format(1.-test_size,test_size,num_of_trials))
ax.set_xlabel('units')
ax.set_ylabel('accuracy')
fig.savefig('results_for_units/result_fig_testsize-{}_ite-{}.png'.format(test_size,iteration))
print('line1:',line1)
print('line2:',line2)
plt.close()


import csv
header = ['index','5','10','15','20','25','30']

with open('results_for_units/result_table_testsize-{}_ite-{}.csv'.format(test_size,iteration),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow(['train']+average_train.tolist())
    writer.writerow(['test']+average_test.tolist())
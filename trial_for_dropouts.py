# coding: utf-8
#=====================================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module  (python2)
#
#   Train with 0% to 40% dropout rate for each layer
#   with 4 hidden layers 10 neurons (fix)
#
#   (c) Keishi Ishihara
#======================================================
from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer
from dropouts import DropoutLayers

# configurations # example
num_of_trials = 10  #5
test_size = 0.2  #0.2
iteration = 25  #25
learning_rate = 0.001  #0.001
units = 10  #10
dropout_rates = [i/100. for i in range(0, 45, 5)]
print('----- hyperparameters ----')
print('num trial {}, num hidden layers {}, train:test={}:{}'.format(num_of_trials,4,1-test_size,test_size))
print('iteration {}, learning_rate {}, units={}'.format(iteration,learning_rate,units))
print ('dropout rates ',dropout_rates)

# load data
X_train, X_test, y_train, y_test = load_data(test_size=test_size)
# load layers
dy = DropoutLayers(units=units)

# arrays for storing results
scores_train = [[0. for i in range(num_of_trials)] for j in range(len(dropout_rates))]
scores_test = [[0. for i in range(num_of_trials)] for j in range(len(dropout_rates))]

for i, dr in enumerate(dropout_rates):
    print('========== Dropout rate: {} ========='.format(dr))
    dy.set_dropouts(dr) # Set dropout rate
    for t in range(num_of_trials):
        print('---- Trial: {} ----'.format(t+1))
        # define model, set params
        clf = Classifier(
            layers=dy.get_layers(),
            learning_rate=learning_rate,
            n_iter=iteration)

        clf.fit(X_train, y_train) # train
        # evaluate
        scores_train[i][t] = clf.score(X_train, y_train)
        scores_test[i][t] = clf.score(X_test, y_test)
        print ('  - Training set score: {}'.format(scores_train[i][t]))
        print ('  - Test set score: {}'.format(scores_test[i][t]))

scores_train = np.array(scores_train)
scores_test = np.array(scores_test)
print('')
print('# Sammary')
print('train:', scores_train)
print('test:', scores_test)
average_train = np.sum(scores_train, axis=1) / float(num_of_trials)
average_test = np.sum(scores_test, axis=1)/ float(num_of_trials)
print ('average_train:', average_train)
print ('average_test:', average_test)


import matplotlib.pyplot as plt
fig = plt.figure() # Initialize Figure
ax = fig.add_subplot(1,1,1) # make axes
x_axis = np.arange(1,len(dropout_rates)+1)
for i in range(num_of_trials): # draw all result with light color
    ax.plot(x_axis, scores_train.T[i], color='blue', linestyle='--', linewidth=0.2)
    ax.plot(x_axis, scores_test.T[i],color='orange', linestyle='--', linewidth=0.3)

line1, = ax.plot(x_axis, average_train, color='blue', label='Average score of training')
line2, = ax.plot(x_axis, average_test, color='orange', label='Average score of training')
ax.legend(loc='best') # add legend
ax.set_xticks(range(x_axis[0],x_axis[-1]+1,1))
ax.set_xticklabels(dropout_rates)
ax.grid(which='both')
ax.set_title('Dropout rate (train:test={}:{}, Trials={})'.format(1.-test_size,test_size,num_of_trials))
ax.set_xlabel('Dropout rate')
ax.set_ylabel('Accuracy')
fig.savefig('results_for_dropout-rates/result_fig_testsize-{}_ite-{}.png'.format(test_size,iteration))
plt.close()


import csv
header = ['index']+dropout_rates
with open('results_for_dropout-rates/result_table_testsize-{}_ite-{}.csv'.format(test_size,iteration),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow(['train']+average_train.tolist())
    writer.writerow(['test']+average_test.tolist())
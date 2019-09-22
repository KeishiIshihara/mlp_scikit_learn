# coding: utf-8
#=============================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module  (python2)
#
#   Train with one layer to ten layers with fixed nodes (10)
#
#   (c) Keishi Ishihara
#=============================================
from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer
from layers import myLayers

# configurations # default
num_of_trials = 5  #5
num_of_layers = 10  #10
test_size = 0.1  #0.2
iteration = 25  #25
learning_rate = 0.001  #0.001
units = 10  #10
print('----- hyperparameters ----')
print('num trial {}, num layers {}, train:test={}:{}'.format(num_of_trials,num_of_layers,1-test_size,test_size))
print('iteration {}, learning_rate {}, units={}\n'.format(iteration,learning_rate,units))

# load data
X_train, X_test, y_train, y_test = load_data(test_size=test_size)
# load layers
ly = myLayers(units=units)
layers_list = ly.get_layers_list()
# arrays for storing results
scores_train = [[0. for i in range(num_of_trials)] for j in range(num_of_layers)]
scores_test = [[0. for i in range(num_of_trials)] for j in range(num_of_layers)]

for l in range(num_of_layers):
    print('========== Num of layers: {} ========='.format(l+1))
    for t in range(num_of_trials):
        print('---- Trial: {} ----'.format(t+1))
        clf = Classifier(
            layers=layers_list[l],
            learning_rate=learning_rate,
            n_iter=iteration)

        clf.fit(X_train, y_train)

        scores_train[l][t] = clf.score(X_train, y_train)
        scores_test[l][t] = clf.score(X_test, y_test)
        print ('  - Training set score: {}'.format(scores_train[l][t]))
        print ('  - Test set score: {}'.format(scores_test[l][t]))


scores_train = np.array(scores_train)
scores_test = np.array(scores_test)
print('')
print('# Sammary')
print('train:', scores_train)
print('test:', scores_test)

average_train = np.sum(scores_train, axis=1) / float(num_of_trials)
average_test = np.sum(scores_test, axis=1)/ float(num_of_trials)
print ('average_train:', average_train) # suppose to be 1xlayers
print ('average_test:', average_test) # suppose to be 1xlayers


import matplotlib.pyplot as plt
fig = plt.figure() # Initialize Figure
ax = fig.add_subplot(1,1,1) # make axes
x_axis = np.arange(1,num_of_layers+1)
for i in range(num_of_trials): # draw all result with light color
    ax.plot(x_axis, scores_train.T[i], color='blue', linestyle='--', linewidth=0.2) # linestyle is not working
    ax.plot(x_axis, scores_test.T[i],color='orange', linestyle='--', linewidth=0.5)

line1, = ax.plot(x_axis, average_train, color='blue', label='Average score of training')
line2, = ax.plot(x_axis, average_test, color='orange', label='Average score of training')
ax.legend(loc='best') # add legend
ax.set_xticks(range(x_axis[0],x_axis[-1]+1,1))
ax.set_xticklabels(x_axis)
ax.grid(which='both')
ax.set_title('Score (train:test={}:{}, Trials={})'.format(1.-test_size,test_size,num_of_trials))
ax.set_xlabel('number of layers')
ax.set_ylabel('accuracy')
fig.savefig('results_for_layers/test_result_fig_testsize-{}_ite-{}.png'.format(test_size,iteration))
plt.close()


import csv
header = ['index']+x_axis.tolist()
with open('results_for_layers/result_table_testsize-{}_ite-{}.csv'.format(test_size,iteration),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow(['train']+average_train.tolist())
    writer.writerow(['test']+average_test.tolist())
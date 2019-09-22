# coding: utf-8
#=====================================================
#  Breast cancer tumor classification
#  with sknn.mlp Classifier module  (python2)
#
#   Train with 2 hidden layers 10 neurons each (fix),
#   adopt 3 different activation function:
#   Rectifier(relu), Sigmoid, Tanh
#
#   (c) Keishi Ishihara
#======================================================
from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_csv_data import load_data
from sknn.mlp import Classifier, Layer
from activations import ActivationLayers

# configurations # example
num_of_trials = 10  #5
test_size = 0.2  #0.2
iteration = 25  #25
learning_rate = 0.001  #0.001
units = 10  #10
activation_list = ['Rectifier', 'Sigmoid', 'Tanh']
print('----- hyperparameters ----')
print('num trial {}, num hidden layers {}, train:test={}:{}'.format(num_of_trials,2,1-test_size,test_size))
print('iteration {}, learning_rate {}, units={}'.format(iteration,learning_rate,units))
print ('activation list ',activation_list)

# load data
X_train, X_test, y_train, y_test = load_data(test_size=test_size)
# load layers
ay = ActivationLayers(units=units)

# arrays for storing results
scores_train = [[0. for i in range(num_of_trials)] for j in range(len(activation_list))]
scores_test = [[0. for i in range(num_of_trials)] for j in range(len(activation_list))]

for i, activation in enumerate(activation_list):
    print('========= Activation func: {} ========='.format(activation))
    ay.set_activation(activation) # Set dropout rate
    for t in range(num_of_trials):
        print('---- Trial: {} ----'.format(t+1))
        # define model, set params
        clf = Classifier(
            layers=ay.get_layers(),
            learning_rate=learning_rate,
            n_iter=iteration)
        # train
        clf.fit(X_train, y_train)
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
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4)) # Initialize Figure
x_axis = np.arange(1,len(activation_list)+1)
# Left fig
axL.violinplot([scores_train.T[0],scores_train.T[1],scores_train.T[2]],showmeans=True)
axL.legend(loc='best') # add legend
axL.set_xticks(range(x_axis[0],x_axis[-1]+1,1))
axL.set_xticklabels(activation_list)
axL.grid(which='both')
axL.set_title('Result of training (train:test={}:{}, Trials={})'.format(1.-test_size,test_size,num_of_trials))
axL.set_xlabel('Activation function')
axL.set_ylabel('Accuracy')
# Right fig
axR.violinplot([scores_test.T[0],scores_test.T[1],scores_test.T[2]],showmeans=True)
axR.legend(loc='best') # add legend
axR.set_xticks(range(x_axis[0],x_axis[-1]+1,1))
axR.set_xticklabels(activation_list)
axR.grid(which='both')
axR.set_title('Result of test (train:test={}:{}, Trials={})'.format(1.-test_size,test_size,num_of_trials))
axR.set_xlabel('Activation function')
axR.set_ylabel('Accuracy')
fig.savefig('results_for_activations/result_fig_testsize-{}_ite-{}.png'.format(test_size,iteration))
plt.close()


import csv
header = ['index']+activation_list
with open('results_for_activations/result_table_testsize-{}_ite-{}.csv'.format(test_size,iteration),'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(header)
    writer.writerow(['train']+average_train.tolist())
    writer.writerow(['test']+average_test.tolist())

# coding: utf-8
#==========================
#  load data from csv
#
#   (c) Keishi Ishihara
#==========================
from __future__ import print_function

from sklearn.model_selection import train_test_split
import numpy as np
import csv

def load_data(base_dir='./', filename='breast-cancer-wisconsin.data.txt', test_size=0.2):
    print('Now loading json files..')
    data = []
    with open(base_dir+filename) as f:
        line = csv.reader(f)
        next(line)
        for row in line:
            data.append(row)

    data = np.array(data)
    data = data[:,1:] # remove id column
    data[data == '?'] = 0 # replace with 0 where the value is ?.
    data = data.astype(np.float16) # cast each data from string to float16
    # train_target_split
    target = data[:,data.shape[1]-1:] 
    data = data[:,:data.shape[1]-1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=0)

    print('data size:')
    print('  - (x_train, y_train) = ({}, {})'.format(len(x_train), len(y_train)))
    print('  - (x_test, y_test) = ({}, {})'.format(len(x_test), len(y_test)))

    return x_train, x_test, y_train, y_test

if __name__=='__main__':
    x_train, y_train, x_test, y_test = load_data()


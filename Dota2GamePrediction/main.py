#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/Users/andreacaliandro/PycharmProjects/hackerrank")

import pandas as pd
from io import StringIO
from sklearn import linear_model, svm
from read_files import read_file, read_stdinput

from transform_dataset import TransformDataset
from modeling import RunningModel

train_df = pd.read_csv('./data/trainingdata.txt',
                       names=['f{}'.format(i) for i in range(10)] + ['winner'])

td = TransformDataset(dataset_df=train_df)
transformed_df = td.get_features().join(td.get_target())

logistic = linear_model.LogisticRegression(C=1)
# svm_model = svm.SVC(kernel='rbf', C=1.0)
ml = RunningModel(transformed_df, model=logistic, target_field='winner', scaling=False)
ml.running_model(0.001)
# ml.model_estimator()


lines = read_stdinput()
num_rows = lines[0].strip()
buf = '\n'.join(lines[1:int(num_rows)+1])
test_df = pd.read_csv(StringIO(unicode(buf)),
                      names=['f{}'.format(i) for i in range(10)] + ['winner'])
td_test = TransformDataset(dataset_df=test_df)
transformed_test_df = td_test.get_features()
y_pred = ml.model.predict(transformed_test_df.as_matrix())
print "\n".join(y_pred.astype(str))


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import linear_model

from modeling import RunningModel
from hackerrank.read_files import json_like_input, read_stdinput, read_file


def main():
    data = read_file('./training-and-test/training.json')
    grades_df = json_like_input(data).fillna(0)
    logistic = linear_model.LogisticRegression(C=10, class_weight='balanced')
    # model = svm.SVC(kernel='rbf', C=1.0, class_weight='balanced')
    ml = RunningModel(grades_df.drop('serial', axis=1),
                      model=logistic,
                      target_field='Mathematics',
                      scaling=False)
    ml.running_model()
    ml.model_estimator()

    data = read_stdinput()
    test_df = json_like_input(data).fillna(0).drop('serial', axis=1)
    features = test_df.as_matrix()
    y_pred = ml.model.predict(features)
    for grade in y_pred:
        print grade

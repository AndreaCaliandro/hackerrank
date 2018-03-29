#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..read_files import read_stdinput, csv_like_input
from modeling import RunningModel


data = read_stdinput()
train_df, test_df = csv_like_input(data, separator=' ')
model =
ml = RunningModel(train_df, test_df, model)
ml.training_model()
ml.price_estimator()
y_pred = ml.predictions()
print "\n".join(y_pred.astype(str))



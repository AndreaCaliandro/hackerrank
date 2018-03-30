#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..read_files import read_file, json_like_input
from modeling import RunningModel
from sklearn import linear_model

data = read_file('./data/training.json')
train_df = json_like_input(data)
train_df.head()

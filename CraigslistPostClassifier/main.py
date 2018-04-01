#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, "/Users/andreacaliandro/PycharmProjects/hackerrank")
for path in sys.path:
    print path

import pandas as pd
import numpy as np
from read_files import read_file, read_stdinput, json_like_input
from bag_of_words import BagOfWords

def train_by_section(train_df):
    ml_dic = {}
    for section in train_df['section'].unique():
        df = train_df[train_df['section']==section]
        ml = BagOfWords(train_docs=df['heading'], train_target=df['category'])
        ml.modeling()
        ml_dic[section] = ml
    return ml_dic


data = read_file('./data/training.json')
train_df = json_like_input(data)
models_dic = train_by_section(train_df)

# test_data = read_stdinput()
test_data = read_file('./data/sample-test.in.json')
test_df = json_like_input(test_data)
for section in test_df['section'].unique():
    test_df.loc[test_df['section'] == section, 'category'] = \
        models_dic[section].predict(test_df[test_df['section'] == section]['heading'])

test_target = read_file('./data/sample-test.out.json')
score = (test_df['category'] == map(str.strip, test_target)).sum()/len(test_target)




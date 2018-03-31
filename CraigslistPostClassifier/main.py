#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, "/Users/andreacaliandro/PycharmProjects/hackerrank")
for path in sys.path:
    print path

from read_files import read_file, json_like_input

data = read_file('./data/training.json')
train_df = json_like_input(data)
train_df.head()

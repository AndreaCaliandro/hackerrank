import pandas as pd
import sys

def read_stdinput():
    return sys.stdin.readlines()

def read_file(filename):
    open_file = open(filename, "r")
    return open_file.readlines()

def json_like_input(lines):
    """
    Create a dataframe from a list of lines, where the first line is an integer N.
    N lines follow each line being a valid JSON object.

    :param lines: list of lines that contains the data
    :return: dataframe
    """
    num_rows = int(lines[0])
    buf =  '[' + ','.join(lines[1:num_rows + 1]) + ']'
    return pd.read_json(buf, orient='records')

def csv_like_input(lines, separator=','):
    num_fields, num_rows = lines[0].split(separator)
    buf = lines[1:int(num_rows)]
    train_df =  pd.read_csv(buf, sep=separator)

    num_rows_2 = lines[int(num_rows)]
    buf = lines[int(num_rows)+1:int(num_rows)+int(num_rows_2)]
    test_df = pd.read_csv(buf, sep=separator)
    return train_df, test_df
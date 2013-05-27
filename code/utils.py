#!/usr/bin/python -tt
import re
import sys

'''
this program houses several utility functions, such as reading in csv files as
training data.

'''

'''
given filename, open file and parse csv file into training data
each row must be in the format:

atrribute_1, attribute_2, ..., attribute_n, label
'''
def parse_training_data(filename):
  T = []
  for line in open(filename, 'r'):
    #remove any whitespace characters
    line = line.strip()
    
    #parse the row, cast each attr as float
    vector = [float(attr.strip()) for attr in line.split(',')]
    #cast label (last element) as an integer (+1 or -1)
    vector[-1] = int(vector[-1])
    #append to training data set
    T.append(vector)
  return T


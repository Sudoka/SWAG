#!/usr/bin/python -tt
import re
import sys
import classifiers
import meta_classifiers
import cPickle as pickle
import math

def main():
  #data = pickle.load(open('../data/data_pickles/bc.pickle', 'r'))
  data = []

  for line in open('../data/fertility/fertility.txt', 'r'):
    tokens = line.strip().split(',')
    v = [float(t) for t in tokens[:-1]]
    if v[0] == -1: v[0] = 0
    elif v[0] == -0.33: v[0] = 1
    elif v[0] == .33: v[0] = 2
    elif v[0] == 1: v[0] = 3

    v[5] += 1
    v[7] += 1
    if tokens[-1] == 'N':
      v.append(-1)
    else:
      v.append(1)
  
    data.append(v)

  test = data[50:]
  data = data[:50]

  C = []
  C.append(classifiers.SVM(data))
  C.append(classifiers.kNN(data))
  C.append(classifiers.NB(data))
  ab = meta_classifiers.AdaBoost(C)
  right = 0.0
  for v in test:
    if ab.classify_vector(v) == v[-1]: right += 1

  print right / len(test)

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


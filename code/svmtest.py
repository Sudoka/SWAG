#!/usr/bin/python -tt
import re
import sys
import classifiers
import cPickle as pickle
from matplotlib import pyplot as plt

def main():
  T = pickle.load(open('../data/data_pickles/pd.pickle', 'r'))
  benchmark(T)


def benchmark(T):
  train = T[len(T)/2:]
  test = T[:len(T)/2]
  y = []
  lims = range(50, len(train), 10)
  for lim in lims:
    svm_cl = classifiers.SVM(train[:lim])
    correct = 0.0
    total = 0
    for t in test:
      if svm_cl.classify_vector(t) == t[-1]:
        correct += 1.0
      total += 1
    y.append(correct/total)
  plt.plot(lims, y)
  plt.show()
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


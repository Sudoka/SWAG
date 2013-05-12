#!/usr/bin/python -tt
import re
import sys
import classifiers
import cPickle as pickle
from matplotlib import pyplot as plt

def main():
  T = pickle.load(open('../data/data_pickles/pd.pickle', 'r'))
  x0, y0 = benchmark(T)
  T = pickle.load(open('../data/data_pickles/bc.pickle', 'r'))
  x1, y1 = benchmark(T)
  T = pickle.load(open('../data/data_pickles/hd.pickle', 'r'))
  x2, y2 = benchmark(T)

  plt.plot(x0, y0)
  plt.plot(x1, y1)
  plt.plot(x2, y2)
  plt.show()

  


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
  return (lims, y)
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


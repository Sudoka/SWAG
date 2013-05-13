#!/usr/bin/python -tt
import re
import sys
import classifiers as cl
import meta_classifiers as mcl
import cPickle as pickle
import math

def main():
  # serialized matrix of heart disease data
  data = pickle.load(open('../data/data_pickles/hd.pickle', 'r'))

  print 'Running classification on', len(data), 'instances of data'
  print 'from Heart Disease dataset\n'

  #split data into two equal sized parts, training data and test data
  lim = len(data)/2
  test = data[lim:]
  data = data[:lim]

  H = mcl.idk_ML(data)

  right = 0.0

  for v in test:
    if H.classify_vector(v) == v[-1]: right += 1.0

  print 'Final accuracy of:', 146.0 / len(test), '\n'

  #see what classifiers we were working with
  for c in H.C:
    print c.get_info(), '\n'

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


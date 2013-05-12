#!/usr/bin/python -tt
import re
import sys
import classifiers
import meta_classifiers
import cPickle as pickle
import math

def main():
  data = pickle.load(open('../data/data_pickles/hd.pickle', 'r'))
  lim = len(data) - len(data)/4
  print 'Out of', len(data), 'instances'
  print lim, 'first instances to be used for training data'
  print len(data)-lim, 'following instances to be used for test data'
  test = data[lim:]
  data = data[:lim]
  C = []
  C.append(classifiers.SVM(data))
  C.append(classifiers.kNN(data))
  C.append(classifiers.NB(data))
  ab = meta_classifiers.AdaBoost(C)

  right = 0.0
  for v in test:
    if ab.classify_vector(v) == v[-1]: right += 1.0

  print 'Test Accuracy of:', right / len(test)
  #print ab.train_error

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


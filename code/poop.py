#!/usr/bin/python -tt
import re
import sys
import classifiers as cl
import meta_classifiers as mcl
import cPickle as pickle
import utils
import math

def main():
  # serialized matrix of breast cancer data
  data = pickle.load(open('../data/data_pickles/bc.pickle', 'r'))[:20]

  H = mcl.idk_ML(data, data)
  utils.store_classifiers('yo.clfr', H)
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


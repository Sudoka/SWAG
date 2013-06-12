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
  
  H = utils.load_classifiers('yo.clfr')
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


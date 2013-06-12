#!/usr/bin/python -tt
import re
import sys
import classifiers as cl
import meta_classifiers as mcl
import cPickle as pickle

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

def store_classifiers(filename, clfr):
  outfile = open(filename, 'wb')
  if clfr.TYPE == 'AdaBoost':
    for c in clfr.C:
      if c.TYPE == 'SVM':
        #saves the svm model as filename-svm.clfr
        c.store_svm(filename.rsplit(',', 1)[0])    
    pickle.dump(clfr, outfile)

  #we are storing a single classifier that is not a SVM, just store like normal
  elif clfr.TYPE != 'SVM':
    pickle.dump(clfr, outfile)

  #we are storing an SVM only
  else:
    clfr.store_svm(filename)
    pickle.dump(clfr, outfile)

  outfile.close()

  

def load_classifiers(filename):
  with open(filename, 'rb') as f:
    clfr = pickle.loads(f.read())
    f.close()

  #remove .clfr
  filename = filename.rsplit('.', 1)[0]

  if clfr.TYPE == 'SVM':
    clfr.load_svm(filename)

  if clfr.TYPE == 'AdaBoost':
    for c in clfr.C:
      if c.TYPE == 'SVM':
        c.load_svm(filename)
  return clfr
  
        

#!/usr/bin/python -tt
import re
import sys
import math
import classifiers

'''
This program houses meta classifiers, such as bagging and boosting algorithms

Like normal classifiers, these meta classifiers will all have the standard
classify_vector and classify_vectors methods, however as parameters to their 
constructors, they will take in ready made classifiers, not data.

Included meta classifiers:

AdaBoost - iteratively determines which classifier should be weighted the highest
  by decreasing weights for inaccurate classifiers each round, if train data given

'''

class AdaBoost:

  def __init__(self, C, val_data=None):
    self.C = C
    #initialize weights of each classifier to be equiprobably
    self.W = [1.0/len(C)]*len(C)

    # training data was specified
    if val_data != None:
      self.optimize_weights(val_data)
    else:
      self.train_error = 'UNKNOWN' 
    
  def optimize_weights(self, val_data):
    #generate validation error for each classifier
    self.validate_classifiers(val_data)

    # generate alpha coeffients
    A = []
    for c in self.C:
      #make it proportial to accuracy
      alpha = 2.718 ** (-.5 * ( math.log(((1.0 - c.validation_error) / c.validation_error) + .001 ) / math.log(2) ))
      A.append(alpha)

    #make sure first iteration runs
    self.validation_error = 0

    counter = 0
    #weighting stabilization usually occurs after 3 iterations
    while counter < 3:
      for i in xrange(len(self.W)):
        self.W[i] = self.W[i]*A[i]
      counter += 1

      self.normalize_weights()
      self.validation_error = self.get_error(val_data)
     

  def normalize_weights(self):
    norm_factor = 0.0
    for w in self.W:
      norm_factor += w
    for i in xrange(len(self.W)):
      self.W[i] = self.W[i] / norm_factor

  def classify_vector(self, v, P=None):
    if sum([self.W[i] * self.C[i].classify_vector(v) for i in xrange(len(self.C))]) < 0:
      return -1
    return 1
    

  #given some data, return the accuracy (error) of this classifier on that data
  def get_error(self, data):
    correct = 0.0
    for v in data:
      if v[-1] == self.classify_vector(v): correct += 1.0
    return correct / len(data)

  def validate_classifiers(self, val_data):
    for c in self.C:
      right = 0.0
      total = len(val_data)
      for v in val_data:
        if v[-1] == c.classify_vector(v): right += 1.0
      c.validation_error = right / total

def idk_ML(data):
  C = []
  C.append(classifiers.SVM(data))
  C.append(classifiers.kNN(data))
  C.append(classifiers.NB(data))
  return AdaBoost(C)

    


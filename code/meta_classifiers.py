#!/usr/bin/python -tt
import re
import sys
import math

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

  def __init__(self, C, train_data=None):
    self.C = C
    self.W = [c.train_error for c in C]
    self.normalize_weights()

    # training data was specified
    if train_data != None:
      self.train_error = self.get_train_error(train_data)
      self.optimize_weights(train_data)
    else:
      self.train_error = 'UNKNOWN' 
    
  def optimize_weights(self, train_data):
    self.train_error = 0

    # generate alpha coeffients
    A = []
    for c in self.C:
      #make it proportial to accuracy
      alpha = 2.718 ** (-.5 * ( math.log(((1.0 - c.train_error) / c.train_error) + .001 ) / math.log(2) ))
      #alpha = alpha * (c.train_error * len(train_data))
      A.append(alpha)


    #keep iterating while there is at least a 1 percent improvement
    while self.get_train_error(train_data) - self.train_error > .001:
      for i in xrange(len(self.W)):
        self.W[i] = self.W[i]*A[i]

      self.normalize_weights()
      self.train_error = self.get_train_error(train_data)
     

  def normalize_weights(self):
    norm_factor = 0.0
    for w in self.W:
      norm_factor += w
    for i in xrange(len(self.W)):
      self.W[i] = self.W[i] / norm_factor

  def classify_vector(self, v):
    if sum([self.W[i] * self.C[i].classify_vector(v) for i in xrange(len(self.C))]) < 0:
      return -1
    return 1
  
  def get_train_error(self, train_data):
    correct = 0.0
    for v in train_data:
      if v[-1] == self.classify_vector(v): correct += 1.0
    return correct / len(train_data)
    


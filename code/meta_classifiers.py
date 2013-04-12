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

AdaBoost - focuses on improving signficance of training data examples
ProBagger - returns a label based on heuristic provided by training accuracy
	    of multiple classifiers
'''

class ProBagger:

  def __init__(classifiers):
    self.C = classifiers

  def classify_vector(self, v):
    score = 0
    for c in self.C:
      if c.classify_vector(v) > 0:
        score += c.plus_accuracy
      else:
        score -= c.minus_accuracy
    if score >= 0
      return 1
    return -1

  def classify_vectors(self, V):
    return [self.classify_vector(v) for v in V]


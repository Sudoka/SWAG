#!/usr/bin/python -tt
import re
import sys
import classifiers
from matplotlib import pyplot as plt

def main():
  data = []
  for line in open('../data/heart-disease/cleaned-heart-disease.data', 'r'):
    data.append([float(num.strip()) for num in line.split(',')])

  #we train our classifier based on the first half of data, iteratively figure out which
  #value of k yields optimal classification results

  x = range(10, len(data)/2, 1)
  y = []

  for i in x:  
    cl1 = classifiers.kNN(data[:i], k=7)
    num_right = 0
    for vector in data[i:]:
      if cl1.classify_vector(vector) == int(vector[-1]):
        num_right += 1
    y.append(float(num_right)/(len(data)-i))

  plt.plot(x, y)
  plt.show()
  
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


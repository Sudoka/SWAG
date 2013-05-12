#!/usr/bin/python -tt
import re
import sys
import classifiers
import cPickle as pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
  data = []
  #for line in open('../data/heart-disease/cleaned-heart-disease.data', 'r'):
    #data.append([float(num.strip()) for num in line.split(',')])

  #we train our classifier based on the first half of data, iteratively figure out which
  #value of k yields optimal classification results

  data = pickle.load(open('../data/data_pickles/pd.pickle', 'r'))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  r = range(30, len(data)/2, 5)
  x = []
  y = []
  z = []
  kvals = [1, 3, 5, 7, 9]

  for _k in kvals:
    for i in r:  
      cl1 = classifiers.kNN(data[:i], k=_k)
      x.append(i)
      y.append(_k)
      z.append(cl1.train_error)

  ax.plot_wireframe(x, y, z)
  #ax.show()
  
  #plt.plot(x, z)
  plt.show()
  
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


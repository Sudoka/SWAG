#!/usr/bin/python -tt
import re
import sys
import classifiers
from matplotlib import pyplot as plt

def main():
  points = []

  #bottom left
  for i in xrange(5):
    for j in xrange(5):
      points.append([i, j, 0])
  
  #bottom right
  for i in xrange(10, 10+5):
    for j in xrange(5):
      points.append([i, j, 0])
  
  #top left
  for i in xrange(5):
    for j in xrange(10, 10+5):
      points.append([i, j, 0])

  #top right
  for i in xrange(10, 10+5):
    for j in xrange(10, 10+5):
      points.append([i, j, 0])

  cl1 = classifiers.kMpp(points, k=4)

  x0 = [p[0] for p in cl1.clusters[0]]
  y0 = [p[1] for p in cl1.clusters[0]]

  x1 = [p[0] for p in cl1.clusters[1]]
  y1 = [p[1] for p in cl1.clusters[1]]

  x2 = [p[0] for p in cl1.clusters[2]]
  y2 = [p[1] for p in cl1.clusters[2]]

  x3 = [p[0] for p in cl1.clusters[3]]
  y3 = [p[1] for p in cl1.clusters[3]]

  plt.plot(x0, y0, 'ro')
  plt.plot(x1, y1, 'bo')
  plt.plot(x2, y2, 'go')
  plt.plot(x3, y3, 'yo')


  
  plt.show()

  
  

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


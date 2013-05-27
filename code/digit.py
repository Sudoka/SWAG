#!/usr/bin/python -tt
import re
import sys
import classifiers
import meta_classifiers
import cPickle as pickle
import math

def main():
  filename = '../data/train_writing.txt'
  f2 = '../data/labels_writing.txt'
  test = []
  
  flag = False
  data = []
  for line in open(filename, 'r'):
    if flag:
      line = line.strip()
      v = []
      for x in line.split(','):
        if x == 'Arabic': x = 1
        if x == 'English': x = 0
        v.append(float(x))
      v[0] = int(v[0])
      data.append(v)
    flag = True

  labels = {}

  for line in open(f2, 'r'):
    line = line.strip()
    tmp = line.split(',')
    labels[int(tmp[0])] = int(tmp[1])

  for i in xrange(len(data)):
    if labels[data[i][0]] > 0:
      data[i].append(1)
    else:
      data[i].append(-1)
  D = data[:]
  data = D[:100]


  flag = False
  for line in open('../data/test.csv', 'r'):
    if flag:
      line = line.strip()
      v = []
      for x in line.split(','):
        if x == 'Arabic': x = 1
        if x == 'English': x = 0
        v.append(float(x))
      v[0] = int(v[0])
      v.append(0)
      test.append(v)
    flag = True


  
  C = []

  C.append(classifiers.SVM(data))
  C.append(classifiers.kNN(data))
  C.append(classifiers.NB(data))
  ab = meta_classifiers.AdaBoost(C)


  counts = {}
  for v in test:
    if v[0] in counts:
      counts[v[0]][1] += 1
      if ab.classify_vector(v) > 0: counts[v[0]][0] += 1.0
    else:
      counts[v[0]] = [0.0, 0]
      if ab.classify_vector(v) > 0: counts[v[0]][0] += 1.0

  f = open('../data/submission.txt', 'w')

  for k in sorted(counts.keys()):
    f.write(str(k) + ',' + str(counts[k][0]/counts[k][1]) + '\n')

  f.close()
  #print ab.train_error


# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


#!/usr/bin/python -tt
import re
import sys
import math
import random
sys.path.append('/home/amrit/SWAG/lib/libsvm/python')
from svmutil import *

'''
This program houses the classes to create classifiers by loading a collection of data vectors and labels

Classifiers present:

kNN - k Nearest Neighbors Classifier
NB - Naive Bayes Classifier
HMM - Hidden Markov Model
kMpp - k Means++

Common Methods to all classifiers:

classify_vector - takes in a vector of data with a label of 0 (unknown) and returns a +1 or -1
  as the predicted label for that vector

classify_data - takes in a set of vectors of data without labels and returns a
  list of predicted labels (+1 or -1) in the same ordering as the set of vectors
'''
class kNN:
  
  '''
  Parameters-

    train_data - collection of vectors of form [v_1, v_2, ... , v_n, label]
    k - integer that signifies number of closest neighbors to look at in classification
    dist - lambda function that specifies the distance metric we will be using

  '''

  def __init__(self, train_data, k=None,):
    self.train_data = train_data
    self.k = k
    self.TYPE = 'kNN'
    
    # hit default case, k is selected from [1, 3, 5, 7, 9]
    if self.k == None:
      self.find_k()
    #k was specified
    else:
      self.train_error = self.get_train_error()
    self.validation_error = 'UNKNOWN'

  def get_info(self):
    s = 'This classifier is a k-Nearest Neighbors classifier'
    s += '\n\tTraining Accuracy: ' + str(self.train_error)
    s += '\n\tChoice of k: ' + str(self.k)

    return s

  def dist(self, x, y):
    num = 0.0
    for i in xrange(len(x)-1):
      num += float(abs(x[i]-y[i])) / ((abs(x[i])+abs(y[i])+.00001)/2)
    return num

  def find_k(self):
    best_k = -1
    best_error = 0.0
    
    # iteratively search through these values of k to find best training error
    for k in [1, 3, 5, 7, 9]:
      self.k = k
      err = self.get_train_error()
      if err > best_error:
        best_k = k
        best_error = err

    # update values of k and train_error accordingly
    self.k = best_k
    self.train_error = best_error


  def get_train_error(self):
    total = len(self.train_data)
    num_right = 0.0

    for i in xrange(total):
      # we must remove the vector itself from the training data, or else it would skew results
      target = self.train_data.pop(i)
      
      if self.classify_vector(target) == target[-1]: num_right += 1.0

      # add vector back
      self.train_data.append(target)

    return num_right/total

  def validate(self, val_data):
    total = len(val_data)
    right = 0.0
    for v in val_data:
      if v[-1] == self.classify_vector(v): right += 1.0
    self.validation_error = right/total

  def classify_vector(self, vector):
    '''
    calculate distances between all training data vectors and the vector to classify
    and stores them along with the index of the corresponding training vector
    then sorts, and returns first k labels, which we then sum to get mode label
    '''

    N = sorted([ ( self.dist(self.train_data[i], vector), self.train_data[i][-1] ) for i in xrange(len(self.train_data))], key=lambda x: x[0])[:self.k]
    label = 0
    for n in N:
      label += n[1]
    if label < 0:
      return -1
    else:
      return 1

  def classify_data(V):
    return [classify_vector(v) for v in V]


class NB:
  

  def __init__(self, train_data):
    self.TYPE = 'NB'

    #create a list of dicts for each dimension of vector belonging to training data
    P = [{} for i in xrange(len(train_data[0])-1)]
    for i in xrange(len(train_data)):
      for j in xrange(len(train_data[0])-1):
        # we want to record the probability of seeing train_data[i][j] at the jth
        # position in given vectors, we arbitrarily choose to record +1 probs
        # -1 probs can be obtained by just taking the complement at the end
        if train_data[i][j] in P[j]:
          if train_data[i][-1] > 0:
            P[j][train_data[i][j]][0] += 1
            P[j][train_data[i][j]][1] += 1
          else:
            P[j][train_data[i][j]][1] += 1
        else:
          if train_data[i][-1] > 0:
            P[j][train_data[i][j]] = [1, 1]
          else:
            P[j][train_data[i][j]] = [0, 1]

    #from final counts, generate probabilities and log them
    for i in xrange(len(P)):
      for k in P[i]:
        if P[i][k][0] != 0:
          P[i][k] = float(P[i][k][0])/float(P[i][k][1])
        else:
          P[i][k] = 1.0/100.0

    self.P = P
    self.train_error = self.get_train_error(train_data)
    self.validation_error = 'UNKNOWN'
    self.binning_threshold = 1.0/100.0


  def get_info(self):
    s = 'This classifier is a Naive Bayes classifier'
    s += '\n\tTraining Accuracy: ' + str(self.train_error)
    s += '\n\tBinning Threshold: ' + str(self.binning_threshold) 

    return s

  def get_train_error(self, train_data):
    total = 0.0
    for vector in train_data:
      if self.classify_vector(vector) == vector[-1]: total += 1
    return total / len(train_data)
      
    

  def classify_vector(self, vector):
    plus = 0
    minus = 0
    # since we are using log scoring, if we do not see a feature, then we say it is 1/100 instead of 0
    #so log(1/100) = -2
    zero = -2
    for i in xrange(len(vector)-1):
      if vector[i] in self.P[i]:
        plus += math.log(self.P[i][vector[i]])
        minus += math.log((1 - self.P[i][vector[i]]) + 1.0/100)

    # return +1 or -1 based on max of plus or minus
    if plus > minus: return 1
    return -1


class SVM:

  def __init__(self, train_data, kernel_type=None, margin=None, gamma=None):
    self.TYPE = 'SVM'
    features = [t[:-1] for t in train_data]
    labels = [t[-1] for t in train_data] 

    prob = svm_problem(labels, features)


    #if kernel was not specified, we must pick which kernel to use
    if kernel_type != 'RBF' and kernel_type != 'Linear':
      #use 2 fold cross validation to determine which kernel to use
      cv_prob = svm_problem(labels[:len(labels)/2], features[:len(features)/2])
      cv_param = svm_parameter('-s 0 -t 2 -q')
      self.model = svm_train(cv_prob, cv_param)
      rbf_acc = self.get_train_error(train_data)

      cv_param = svm_parameter('-s 0 -t 0 -q')
      self.model = svm_train(cv_prob, cv_param)
      lin_acc = self.get_train_error(train_data)

      if rbf_acc > lin_acc:
        kernel_type = 'RBF'
      else:
        kernel_type = 'Linear'

    if kernel_type == 'RBF':
      self.kernel_function = kernel_type
      if margin == None and gamma == None:      
        self.rbf_grid_search(train_data, '-s 0 -t 2 -q', prob)
      if margin == None and gamma != None:
        self.linear_margin_search(train_data, '-s 0 -t 2 -q', prob)
      if margin != None and gamma == None:
        self.linear_gamma_search(train_data, '-s 0 -t 2 -q', prob)

      if margin != None and gamma != None:
        self.margin = margin
        self.gamma = gamma
        param = svm_parameter('-s 0 -t 0 -q -g ' + str(self.gamma) + ' -c ' + str(self.margin))
        self.model = svm_train(prob, param)
 

    elif kernel_type == 'Linear':
      self.kernel_function = kernel_type
      if margin == None:
        self.linear_margin_search(train_data, '-s 0 -t 0 -q', prob)
      else:
        self.margin = margin
        param = svm_parameter('-s 0 -t 0 -q -c ' + str(self.margin))
        self.model = svm_train(prob, param)

    self.train_error = self.get_train_error(train_data)

  def get_info(self):
    s = 'This classifier is a SVM'
    s += '\n\tTraining Accuracy: '+ str(self.train_error)
    s += '\n\tKernel Function: ' + self.kernel_function
    s += '\n\tMargin Cost: ' + str(self.margin)
    #gamma coefficient only exists for radial basis function
    if self.kernel_function == 'RBF':
      s += '\n\tGamma Coefficient: ' + str(self.gamma)

    return s
    

  def classify_vector(self, v):
    p_labels, p_acc, p_vals = svm_predict([0], [v[:-1]], self.model, '-q')
    return int(p_labels[0])

  def classify_data(self, V):
    return svm_predict([0]*len(V), [v[:-1] for v in V], self.model)[0]

  def normalize_vector(self, v):
    return [(v[i]-self.min_attrs[i])/(self.max_attrs[i]-self.min_attrs[i] + .00001) for i in xrange(len(v)-1)] + [v[-1]]

  def get_train_error(self, train_data):
    total = 0.0
    for vector in train_data:
      if vector[-1] == self.classify_vector(vector): total += 1.0
    return total / len(train_data)

  def linear_margin_search(self, train_data, p_string, prob):
    C = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
    best_error = 0.0
    best_c = -1
    for c in C:
      param = svm_parameter(p_string + ' -c ' + str(c))
      self.model = svm_train(prob, param)
      error = self.get_train_error(train_data)
      if error > best_error:
        best_error = error
        best_c = c
        best_param = param

    self.model = svm_train(prob, param)
    self.margin = best_c
    self.train_error = best_error

  def linear_gamma_search(self, train_data, p_string, prob):
    G = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
    p_string += '-c ' + str(self.margin)
    best_error = 0.0
    best_g = -1
    for g in G:
      param = svm_parameter(p_string + ' -g ' + str(g))
      self.model = svm_train(prob, param)
      error = self.get_train_error(train_data)
      if error > best_error:
        best_error = error
        best_g = g
        best_param = param

    self.model = svm_train(prob, param)
    self.gamma = best_g
    self.train_error = best_error



  #a grid search to find optimal values of margin, c, and gamma, g
  def rbf_grid_search(self, train_data, p_string, prob):
  
    #values we will search out for optimal c and g
    C = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
    G = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]

    #max error
    max_err = 0.0
    #init param
    best_param = svm_parameter(p_string)
    best_c = 0
    best_g = 0

    for i in xrange(len(C)):
      for j in xrange(len(G)):
        param = svm_parameter(p_string + ' -c ' + str(C[i]) + ' -g ' + str(G[j]))
        self.model = svm_train(prob, param)
        error = self.get_train_error(train_data)
        if error > max_err:
          max_err = error
          best_param = param
          best_c = C[i]
          best_g = G[j]

    # we relax the margin by a factor of two to deal with overfitting
    best_param = p_string + ' -c ' + str(2*best_c) + ' -g ' + str(best_g)
    self.model = svm_train(prob, best_param)
    self.margin = 2*best_c
    self.gamma = best_g
    self.train_error = max_err

  #returns a string that will be serialized later
  def store_svm(self, filename):
    svm_save_model(filename + '-svm.clfr', self.model)
    self.model = None
  
  def load_svm(self, filename):
    self.model = svm_load_model(filename + '-svm.clfr')
    

class kMpp:
  '''
  Parameters:
    data	- vectors of data    
    k		- number of clusters, by default it is set to sqrt(len(data))
    dist_met	- distance metric to be used, by default euclidean is used
    iter_max	- maximum amount of iterations to be used when converging
    		  clusters, by default this is set to 1024
  '''

  def __init__(self, data, k=None, dist_met=None, iter_max=128):
    self.TYPE = 'kMpp'
    if k == None:
      k = int(len(data)**.5)

    #by default use relative dist metric
    if dist_met == None:
      dist = lambda x, y: sum([float(abs(x[i]-y[i])) / ((abs(x[i])+abs(y[i]) + .0001)/2) for i in xrange(len(x)-1)])
    elif dist_met == 'euclidean':
      dist = lambda x, y: sum([(x[i]*y[i])**2 for i in xrange(len(x)-1)])**.5
    elif dist_met == 'manhattan':
      dist = lambda x, y: sum([abs(x[i]-y[i]) for i in xrange(len(x)-1)])


    # this list will house clusters as elements of this list, with each cluster
    # being a list of vectors, the first element being the cluster center, and
    # following elements being a part of that cluster
    self.clusters = self.converge_clusters(self.seed_clusters(k, data, dist), iter_max, dist, data)
    self.k = k
    self.iter_max = iter_max

  # seeds cluster centers as per the kMeans++ algorithm
  def seed_clusters(self, k, data, dist):    
    # Randomly pick the first cluster center, and since only one cluster center
    # exists at this point, all data vectors must be belong to this cluster
    clusters = [[random.choice(data)] + data[:]]

    for i in xrange(1, k):
      cluster_dists = []
      for cluster in clusters:
        for i in xrange(1, len(cluster)):
          cluster_dists.append((cluster[0], dist(cluster[0], cluster[i])**2))

      clusters = [[cluster[0]] for cluster in clusters]
      clusters.append([weighted_choice(cluster_dists)])
      
      # update clusters
      clusters = update_clusters(dist, data, clusters)
    return clusters

  def converge_clusters(self, clusters, iter_max, dist, data):
    im = iter_max-1
    prev_centers =  [cluster[0] for cluster in clusters]

    # update the cluster centers as the mean vector of cluster members and then
    # updates clusters with the new centers
    clusters = update_clusters(dist, data, update_centers(clusters))

    # will keep updating cluster until iter_max has been hit
    while im > 0:
      clusters = update_clusters(dist, data, update_centers(clusters))
      im -= 1

    return clusters
      
      

def average_vector(vectors):
  avg_vect = [0]*(len(vectors[0]))
  for vector in vectors:
    for i in xrange(len(vector)-1):
      avg_vect[i] += vector[i]
  return [avg_vect[i]/float(len(vectors)) for i in xrange(len(avg_vect)-1)]+[0]

def update_centers(clusters):
  for i in xrange(len(clusters)):
    clusters[i][0] = average_vector(clusters[i][:])
  return clusters


def update_clusters(dist, data, clusters):
  new_clusters = []
  for cluster in clusters:
    new_clusters.append([cluster[0]])
  for vector in data:
    # chooses the closest cluster center and appends the vector to that cluster
    idx = closest_center(vector, clusters, dist)
    new_clusters[idx].append(vector)
  return new_clusters
    

def closest_center(vector, clusters, dist):
  return min([(dist(vector, clusters[i][0]), i)for i in xrange(0, len(clusters))], key=lambda x: x[0])[1]
  
def weighted_choice(items):
  weight_total = sum((item[1] for item in items))
  n = random.uniform(0, weight_total)
  for item, weight in items:
      if n < weight:
          return item
      n = n - weight
  return item

class HMM:
  '''
  this class contains exactly 3 dictionary objects:
    start_p - contains keys (representing markov states) referencing the initial probabilities required to get to a specific state
            if this is unknown, simple assign each state probability 1/n, with n = amount of states

    state_p - contains keys (representing markov states) referencing the probabilities of changing to another state, or staying
            at the same state

    hidden_p - contains keys (representing hidden states) referencing the probabilities of getting to a specific hidden state from any markov state


  This class also contains specific methods, such as:
    viterbi_solve(obs) - takes in a sequence of observations and attempts to generate the most likely sequency of markov states corresponding to
                       the given hidden states. Uses DP, solves in time O((n+m)^2), where n = no. of markov states, and m = no. of hidden states
                       this algorithm is a definite improvement of just brute-force checking every solution, which would be O(2^n)



  In all examples, uppercase letters A B C reference markov states, whereas lowercase letters a b c reference hidden states
  '''

  def __init__(self):
    self.TYPE = 'HMM'
    self.start_p = {}
    self.state_p = {}
    self.obs_p = {}
    self.states = []

  #should be fed a dict of form {'A': .4, 'B': .3, 'C': .3}
  def set_start_p(self, start_p):
    self.start_p = start_p

  #should be fed a dict of form {'A': {'A': .1, 'B': .2, 'C': .7}, 'B' : {'A': .5, 'B': .4, 'C': .1}, 'C' : {'A': .2, 'B': .6, 'C': .2}}
  #essentially the dict should have each state as a key, and each of those keys should have, as values, dicts that have the probabilities associated
  #with switching to the other states or staying at the current state
  def set_state_p(self, state_p):
    self.state_p = state_p
    self.states = state_p.keys()

  #should be fed a dict of form {'A': {'a': .9, 'b': .05, 'c': .05}, 'B' : {'a': .4, 'b': .2, 'c': .4}, 'C' : {'a': .2, 'b': .7, 'c': .1}}
  #essentially the dict has, as keys, the markov states, and the probability to get to the hidden state from each of the markov states.
  def set_hidden_p(self, hidden_p):
    self.hidden_p = hidden_p

  #returns a list of the most probable markov states associated with the given hidden states
  #parameters:
  #  obs - a list of 'observations' aka a sequence of hiddens states we have observed
  def viterbi_solve(self, obs):
    path = []
    tmp_path = {}
    #since the first state we want to find is the only one that start_p, we will handle that seperately.
    for k in self.states:
      #we generate the probs associated with choosing every markov state as a possible path
      tmp_path[k] = self.start_p[k]*self.hidden_p[k][obs[0]]
    path_max = max(tmp_path, key=tmp_path.get)
    #most probable markov state, along with the corresponding prob
    path.append((path_max, tmp_path[path_max]))
    for i in xrange(1, len(obs)):
      curr_state = path[-1][0]
      tmp_path = {}
      #generate all possible paths originating from the current state we are at
      for k in self.states:
        tmp_path[k] = self.state_p[curr_state][k]*self.hidden_p[k][obs[i]]
      #take max probability from the states
      path_max = max(tmp_path, key=tmp_path.get)
      
      #since the probability of reach any given state in the path is the prob of the previous state in the path time the p(path_max)
      #we adjust for this
      path.append((path_max, tmp_path[path_max]*path[-1][1]))
    return path

'''
given a data distribution D, normalize each attribute (except last element, that
is reserved for labels). normalize value x from attribute a as follows:
norm_x_a = x_a-min_a/(max_a-min_a)

Returns:
modified data distribution that is normalized
list containing maximum attribute values
list containing minimum attribute values
'''
def normalize_data(D):
  _D = []

  #init values
  max_attrs = D[0]
  min_attrs = D[0]
  
  #find max and min attribute values
  for d in D:
    #skip over labels
    for i in xrange(len(d)-1):
      if d[i] < min_attrs[i]: min_attrs[i] = d[i]
      if d[i] > max_attrs[i]: max_attrs[i] = d[i]

  for i in xrange(len(D)):
    for j in xrange(len(D[0])-1):
      #normalize
      D[i][j] = ( D[i][j]-min_attrs[j] ) / ( max_attrs[j]-min_attrs[j] + .00001 )

  return (D, max_attrs, min_attrs)
    

#!/usr/bin/python -tt
import re
import sys
import math

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

  def __init__(self, train_data, k, dist=None):
    self.vectors = vectors
    self.k = k
    
    #default distance metric is relative difference, not euclidean
    #this is to make ensure that different types of data can be examined in the same vector
    if dist == None:
      self.dist = lambda x, y: sum([float(abs(x[i]-y[i])) / ((abs(x[i])+abs(y[i]))/2) for i in xrange(len(x)-1)])
    else:
      self.dist = dist

  def classify_vector(vector):
    #calculate distances between all training data vectors and the vector to classify
    #and stores them along with the index of the corresponding training vector
    #then sorts, and returns first k labels
    N = sorted([ ( self.dist(train_data[i], vector), i ) for i in xrange(len(train_data))], key=lambda x: x[0])[:self.k]
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
            P[train_data[i][j]][1] += 1
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

  def classify_vector(self, vector):
    plus = 0
    minus = 0
    # since we are using log scoring, if we do not see a feature, then we say it is 1/100 instead of 0
    #so log(1/100) = -2
    zero = -2
    for i in xrange(len(vector)-1):
      if vector[i] in self.P[i]:
        plus += math.log(self.P[i][vector[i]])
        minus += math.log(1 - self.P[i][vector[i]])

    # return +1 or -1 based on max of plus or minus
    if plus > minus: return 1
    return -1

class kMpp:
  '''
  Parameters:
    data	- vectors of data    
    k		- number of clusters, by default it is set to sqrt(len(data))
    dist_met	- distance metric to be used, by default euclidean is used
    iter_max	- maximum amount of iterations to be used when converging
    		  clusters, by default this is set to 1024
  '''
  def __init__(self, data, k=None, dist_met=None, iter_max=1024):
    if k == None:
      k = int(sqrt(len(data)))
    if dist_met == None or 'euclidean':
      dist = lambda x, y: sum([(x[i]*y[i])**2 for i in xrange(len(x))])**.5
    elif dist_met == 'manhattan':
      dist = lambda x, y: sum([abs(x[i]-y[i]) for i in xrange(len(x))])


    # this list will house clusters as elements of this list, with each cluster
    # being a list of vectors, the first element being the cluster center, and
    # following elements being a part of that cluster
    self.clusters = converge_clusters(seed_clusters(k, data, dist), iter_max, dist, data)
    self.k = k
    self.iter_max = iter_max

  # seeds cluster centers as per the kMeans++ algorithm
  def seed_clusters(k, data, dist):    
    # Randomly pick the first cluster center, and since only one cluster center
    # exists at this point, all data vectors must be belong to this cluster
    clusters = [random.choice(data), data[:]]

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

  def converge_clusters(clusters, iter_max, dist, data):
    im = iter_max-1
    prev_centers =  [cluster[0] for cluster in clusters]

    # update the cluster centers as the mean vector of cluster members and then
    # updates clusters with the new centers
    clusters = update_clusters(dist, data, update_centers(clusters))

    # will keep updating cluster until iter_max has been hit or clusters converge
    while im > 0 and [cluster[0] for cluster in clusters] != prev_centers:
      prev_centers = [cluster[0] for cluster in clusters]
      clusters = update_clusters(dist, data, update_centers(clusters))
      im -= 1

  return clusters
      
      

  def average_vector(vectors):
    avg_vect = [0]*len(vectors[0])
    for vector in vectors:
      for i in xrange(len(vector):
        avg_vect[i] += vector[i]
    return [avg_vect[i]/float(len(vectors)) for i in xrange(len(avg_vect))]

  def update_centers(clusters):
    for i in xrange(len(clusters)):
      clusters[i][0] = average_vector(clusters[i][1:])
    return clusters


  def update_clusters(dist, data, clusters):
    for vector in data:
      # chooses the closest cluster center and appends the vector to that cluster
      clusters[closest_center(vector, cluster, dist)].append(vector)
    return clusters
    

  def closest_center(vector, clusters, dist):
    return min([(dist(vector, cluster[i][0]), i)for i in xrange(0, len(clusters))], key=lambda x: x[0])[1]
  
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

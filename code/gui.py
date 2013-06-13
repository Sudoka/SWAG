#!/usr/bin/python -tt
import re
import sys
import classifiers as cl
import meta_classifiers as mcl
import cPickle as pickle
import easygui as eg
import utils

def main():
  reply = eg.boolbox(msg='Hello, do you have a pre-computed classifier?', choices=('Yes', 'No'))
  print reply
  if reply == 1:
    filename = eg.fileopenbox(msg='Please specificy cached classifier file ending in .clfr')
    model = utils.load_classifiers(filename)
    reply = eg.boolbox(msg='Now that we have specified our classifier, we must now specify data to be classified. Are you ready to proceed?', choices=('Yes', 'No'))
    if reply == 0:
      sys.exit()
    filename = eg.fileopenbox(msg='Please specify data to be classified.')
    D = utils.parse_data(filename)
    outfilename = ''.join(filename.split('.')[:-1]) + '.lbls'
    with open(outfilename, 'w') as f:
      for d in D:
        f.write(str(model.classify_vector(d)))
        f.write('\n')
  else:
    filename = eg.fileopenbox(msg='Please specify training data for your classifier')
    D = utils.parse_data(filename)
    reply = eg.boolbox(msg='Would you like to engage in manual or automatic construction of your classifier?', choices=('Manual', 'Automatic'))
    #manual selection
    if reply == 1:
      algs = eg.multchoicebox(msg='Please choose at least one algorithm:', choices=('k-Nearest Neighbors', 'Support Vector Machine', 'Naive Bayes'))
      alg_params = {alg : 'auto' for alg in algs}
      #storage for set of classifiers
      C = []
      for alg in algs:
        reply = eg.boolbox(msg='Would you like to engage in manual or automatic parameter tuning for your ' + alg + ' algorithm?', choices=('Manual', 'Automatic'))
        if reply == 1:
          if alg[0] == 'k':
            params = eg.multenterbox(msg='Please select the following parameters for your ' + alg + ' algorithm:', fields=('k'), values=['1'])
            C.append(cl.kNN(D, k=int(params[0])))
          if alg[0] == 'S':
            reply = eg.boolbox(msg='What type of kernel would you like to use for your Support Vector Machine?', choices=('Radial Basis Function', 'Linear'))
            if reply == 1:
              params = eg.multenterbox(msg='Please select the following parameters for your RBF Support Vector Machine:', fields=('margin', 'gamma'), values=['1.0', '1.0'])
              C.append(cl.SVM(D, kernel_type='RBF', margin=float(params[0]), gamma=float(params[1])))
            else:
              params = eg.multenterbox(msg='Please select the following parameters for your Linear Support Vector Machine:', fields=('margin'), values=['1.0'])
              C.append(cl.SVM(D, kernel_type='Linear', margin=float(params[0])))
          if alg[0] == 'N':
            params = eg.multenterbox(msg='Please select the following parameters for your Naive Bayes algorithm:', fields=('bin threshold'), values=['.1'])
            C.append(cl.NB(D))

        else:
          if alg[0] == 'k':
            C.append(cl.kNN(D))
          if alg[0] == 'S':
            C.append(cl.SVM(D))
          if alg[0] == 'N':
            C.append(cl.NB(D))

      model = mcl.AdaBoost(C)

    #automatic selection
    else:
      model = mcl.AdaBoost([cl.kNN(D), cl.SVM(D), cl.NB(D)])

    reply = eg.boolbox(msg='Now that we have specified our classifier, we must now specify data to be classified. Are you ready to proceed?', choices=('Yes', 'No'))
    if reply == 0:
      sys.exit()
    filename = eg.fileopenbox(msg='Please specify data to be classified.')
    D = utils.parse_data(filename)
    outfilename = ''.join(filename.split('.')[:-1])
    with open(outfilename + '.lbls', 'w') as f:
      for d in D:
        f.write(str(model.classify_vector(d)))
        f.write('\n')
    #cache our classifier
    utils.store_classifiers(outfilename, model)
    #give user some information on classifiers used
    open(outfilename + '.info', 'w').write(model.get_info())


      
      
# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()


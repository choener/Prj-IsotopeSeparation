#!/usr/bin/env python3

from os.path import exists
from pymc import Model, Normal, HalfCauchy, sample, Dirichlet, HalfNormal
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import aesara as ae
import aesara.tensor as at
import argparse
import arviz as az
import matplotlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import pickle
import pymc as mc

import Construct

font = { 'weight': 'bold', 'size': 30 }
pl.rc('font', **font)

# Histogram of the relative nucleotide compositions, divided by label. We want to make sure that we
# don't accidentally condition on the nucleotide composition, instead of on the deuterium
# composition.

def nucleotideHistogram (pd):
  # TODO plot n1rel histogram, but separate for each label !
  pass

# TODO run a simple mono- and dinucleotide model for the mean of the signal. Do this for
# raw data
# z-score data
# EMA data
# beta * [A-count,C-count,G-count,T-count,1] ~ observed mean
# now, we need to add the change due to deuterium, which might be a shift in beta?

def modelDirichletNucs (pd):
  with Model():
    ys = np.array(pd['means'])
    ls = np.array(pd['labels'])
    #
    xs1 = np.asmatrix(np.vstack(pd['n1rel'].values))
    _, cols1 = xs1.shape
    print(cols1)
    baseDisp1  = Dirichlet('bs dsp 1', np.ones(cols1))
    baseScale1 = Normal('bs scl 1', 0, sigma=3)
    deutDisp1  = Dirichlet('dt dsp 1', np.ones(cols1))
    deutScale1 = Normal('dt scl 1', 0, sigma=3)
    mu = np.mean(pd['means'])
    mu += baseScale1 * at.dot(xs1,baseDisp1)
    mu += deutScale1 * at.dot(xs1,deutDisp1) * ls
    #
    #xs2 = Data('xs2', value = xsMat2)
    #rows2, cols2 = xsMat2.shape
    #baseDisp2  = Dirichlet('bs dsp 2', np.ones(cols2))
    #baseScale2 = Normal('bs scl 2', 0, sd=3)
    #deutDisp2  = Dirichlet('dt dsp 2', np.ones(cols2))
    #deutScale2 = Normal('dt scl 2', 0, sd=3)
    #mu += baseScale2 * tt.dot(xs2,baseDisp2)
    #mu += deutScale2 * tt.dot(xs2,deutDisp2) * ls
    #
    epsilon = HalfCauchy('ε', 5)
    likelihood = Normal('ys', mu = mu, sigma = epsilon, observed = ys)
    trace = sample(1000, return_inferencedata = True, init="adapt_diag")
    #traceDF = trace_to_dataframe(trace)
    #print(traceDF.describe())
    #print(traceDF['bs scl 1'] < traceDF['dt scl 1'])
    #scatter_matrix(traceDF, figsize=(8,8))
    #traceplot(trace)
    az.plot_posterior(trace)
    plt.savefig('posterior.pdf', bbox_inches='tight')
    az.plot_trace(trace)
    plt.savefig('trace.pdf', bbox_inches='tight')
    # TODO extract the full trace so that I can run a prob that deutscale != 0
    # TODO extract statistics on nucleotide distribution, compare between classes to make sure we
    # don't accidentally train just on that
    #prob_diff = np.mean(trace[:]['bs scl 1'] < trace[:]['dt scl 1'])
    #print('P(mean_base < mean_deut) = %.2f%%' % (prob_diff * 100))

# Normalizes all pandas data, return the new pd frame and the mean,sigma.
# NOTE we normalize on canonical data to make the difference more clear

def normalize(pdd):
  pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  mean = pd['means'].mean() # pd[pd['labels']==0]['means'].mean()
  stddev = pd['means'].std()
  pd['means'] = (pd['means'] - mean) / stddev
  print(len(pd))
  print(len(pd[pd['labels']==0]))
  return pd, mean, stddev

# NOTE a variant with 3-mers is possible but leads to divergence of the sampler somewhat often

def modelMixtureNucs(pdAll, k):
  pd, test = pdSplit(pdAll)
  ys = np.array(pd['means'])
  ls = np.array(pd['labels'])
  xs1 = np.asmatrix(np.vstack(pd['n1rel'].values))
  xs2 = np.asmatrix(np.vstack(pd['n2rel'].values))
  xs3 = np.asmatrix(np.vstack(pd['n3rel'].values))
  _, cols1 = xs1.shape
  _, cols2 = xs2.shape
  _, cols3 = xs3.shape
  xs = None
  if k==1:
    xs = xs1
  if k==2:
    xs = xs2
  if k==3:
    xs = xs3
  rows, cols = xs.shape
  with Model() as model:
    beta = Normal('β', mu = np.zeros(cols), sigma = 1, shape=(cols))
    #deut = Normal('δ', mu = np.zeros(cols), sigma = 1, shape=(cols))
    deut = Normal('δ', mu = 0, sigma = 1) # , shape=(1)) # make this scalar (again)
    ll  = at.dot(4*xs-1, beta)
    ll += deut * ls
    error = HalfCauchy('ε', beta = 1)
    likelihood = Normal('ys', mu = ll, sigma = error, observed = ys,shape=(rows))
    trace = memoize(f'mixture-{k}.model', model)
  print('sampling finished')
  #
  varNames = ['ε', 'β', 'δ']
  az.plot_trace(trace,figsize=(20,20))
  plt.savefig(f'mixture-{k}-trace.pdf', bbox_inches='tight')
  az.plot_posterior(trace, var_names = varNames,figsize=(30,20), textsize=26)
  plt.savefig(f'mixture-{k}-posterior.pdf', bbox_inches='tight')
  #
  s = az.summary(trace, var_names = varNames)
  print(s)
  #
  # plot on y=0 all samples with label 0, y=1, label 1; x-axis should have adjusted mu
  #
  return {} # return { 'mu': mu, 'beta1': beta1, 'deut': deut, 'err': error }

# 
#
# NOTE Reading 'reads' is costly only the first run, we pickles immediately, then re-use the pickles

def main ():
  print(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('--reads', action='append', help='directories where reads are located')
  parser.add_argument('--pickle', default="./tmp", help='where to write pickle data to')
  parser.add_argument('--limitreads', help='Limit the number of reads to read when no pickle exists')
  parser.add_argument('--plotsquiggle', help='Plot the time series squiggle plot for every (!) read')
  args = parser.parse_args()
  print(args)
  # fill infrastructure for data
  construct = Construct.Construct(barcodes = args.barcode, reads = args.reads, pickleDir = args.pickle, limitReads = args.limitreads, plotSquiggle = args.plotsquiggle)
  #labels={}
  #labels['0'] = getIdLabels('barcode14.ids')
  #labels['30'] = getIdLabels('barcode15.ids')
  #labels['100'] = getIdLabels('barcode16.ids')
  #dir = '/shared/choener/Workdata/heavy_atoms_deuterium_taubert/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #dir = '.'
  #dir = '/data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #pdAll = None
  #if exists('./pdAll.pandas'):
  #  pdAll = pandas.read_pickle('./pdAll.pandas')
  #else:
  #  pds = []
  #  cnt = 0
  #  for path in Path(dir).rglob(f'*.fast5'):
  #    pd = fast5Handler(labels,path)
  #    pds.append(pd)
  #    cnt += 1
  #    #if cnt >= 10: break
  #  allpds = pandas.concat(pds)
  #  allpds.to_pickle('./pdAll.pandas')
  #  pdAll = allpds
  #relNucComp(pdAll)
  #nucleotideHistogram(pdAll)
  ## only work with data that has known labels
  #pdd = pdAll[pdAll['labels'] != -1]
  #pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  #pd, pdmean, pdstddev = normalize(pdd)
  ## TODO split off train / test
  #print(len(pd))
  #pd = pd.sample(frac=1)
  ##modelDirichletNucs(pd)
  #modelMixtureNucs(pd, k=1)
  #modelMixtureNucs(pd, k=2)
  #modelLogistic(pd, k=1)
  #modelLogistic(pd, k=2)
  ##testModel (pd[splitPoint:], mdl)

if __name__ == "__main__":
  main()

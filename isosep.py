#!/usr/bin/env python3

from collections import Counter
from itertools import chain
from os.path import exists
from pathlib import Path
from pymc import Model, Normal, HalfCauchy, sample, Dirichlet, HalfNormal
import aesara as ae
import aesara.tensor as at
import arviz as az
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import pymc as mc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

# Jannes has extracted the IDs which correspond to 0%, 30%, 100% deuterium. However, for now we only
# care about 0% vs 100% deuterium

def getIdLabels(fname):
  labels=set()
  with open(fname, 'r') as r:
    for line in r:
      labels.add(line.strip())
  return labels

# Store both k=1 and k=2 statistics

class NucleotideStats:
  def __init__(self, seq):
    s2 = [''.join(pair) for pair in zip(seq[:-1],seq[1:])]
    s3 = [''.join(triplet) for triplet in zip(seq[:-2],seq[1:-1],seq[2:])]
    self.k1 = Counter(chain.from_iterable(seq))
    self.k2 = Counter(s2)
    self.k3 = Counter(s3)

#

def maximalRunOf(fast5):
  rs = []
  for k in fast5['Analyses/'].keys():
    if 'Segmentation' in k:
      rs.append(int(k.split('_')[1]))
  return max(rs)

#
# TODO Make this generic over the k-mers

def fast5Handler (labels,fname):
  print(fname)
  fast5 = h5py.File(fname, 'r')
  ns = []
  ms = []
  ls = []
  i=0
  for r in fast5.keys():
    i+=1
    if i>=100: break
    s = maximalRunOf(fast5[r])
    rid = r.split('read_')[1]
    rawSignal = fast5[r]['Raw/Signal'][:]
    start = fast5[r][f'Analyses/Segmentation_{s:03d}/Summary/segmentation'].attrs['first_sample_template']
    fromStart = rawSignal[start:]
    fastq = fast5[r][f'Analyses/Basecall_1D_{s:03d}/BaseCalled_template/Fastq'][()].decode('utf-8').split('\n')[1]
    ch = fast5[r][f'channel_id']
    digitisation = ch.attrs['digitisation']
    offset = ch.attrs['offset']
    range = ch.attrs['range']
    scale = range / digitisation
    pA = scale * (fromStart + offset)
    nstats = NucleotideStats(fastq)
    ns.append(nstats)
    ms.append(np.mean(pA))
    for l in labels:
      if rid in l:
        ls.append(l)
        break
    if rid in labels['0']:
      ls.append(0)
    elif rid in labels['100']:
      ls.append(1)
    elif rid in labels['30']:
      ls.append(2)
    else:
      ls.append(-1)
  fast5.close()
  print(len(ns))
  # TODO construct data frame, where we can then later cut things away
  return pandas.DataFrame(
    { 'nstats': ns
    , 'means': ms
    , 'labels': ls
    })

def relNucComp(pd):
  ks1 = set()
  ks2 = set()
  ks3 = set()
  for n in pd['nstats']:
    for k in n.k1.keys():
      ks1.add(k)
    for k in n.k2.keys():
      ks2.add(k)
    for k in n.k3.keys():
      ks3.add(k)
  ks1=list(ks1)
  ks1.sort()
  ks2=list(ks2)
  ks2.sort()
  ks3=list(ks3)
  ks3.sort()
  print(ks1)
  print(ks2)
  print(ks3)
  xs1 = []
  xs2 = []
  xs3 = []
  for n in pd['nstats']:
    s1 = sum(n.k1.values())
    s2 = sum(n.k2.values())
    s3 = sum(n.k3.values())
    arr1 = np.array([n.k1[k] / s1 for k in ks1])
    arr2 = np.array([n.k2[k] / s2 for k in ks2])
    arr3 = np.array([n.k3[k] / s3 for k in ks3])
    xs1.append(arr1)
    xs2.append(arr2)
    xs3.append(arr3)
  pd['n1rel'] = xs1
  pd['n2rel'] = xs2
  pd['n3rel'] = xs3
  xs1bs = np.vstack(pd['n1rel'][pd['labels']==0])
  xs1dt = np.vstack(pd['n1rel'][pd['labels']==1])
  _, axes = plt.subplots(2,4, figsize=(40,10))
  for c,lbl in enumerate(ks1):
    ax = axes[0,c]
    az.plot_posterior({lbl + ' base': xs1bs[:,c]},ax=ax, rope=(0,1))
    ax = axes[1,c]
    az.plot_posterior({lbl + ' deut': xs1dt[:,c]},ax=ax, rope=(0,1))
  plt.savefig('nucleotide-distributions.pdf')
  return

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
    plt.savefig('posterior.pdf')
    az.plot_trace(trace)
    plt.savefig('trace.pdf')
    # TODO extract the full trace so that I can run a prob that deutscale != 0
    # TODO extract statistics on nucleotide distribution, compare between classes to make sure we
    # don't accidentally train just on that
    #prob_diff = np.mean(trace[:]['bs scl 1'] < trace[:]['dt scl 1'])
    #print('P(mean_base < mean_deut) = %.2f%%' % (prob_diff * 100))

# Normalizes all pandas data, return the new pd frame and the mean,sigma.
# NOTE we normalize on canonical data to make the difference more clear

def normalize(pdd):
  pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  mean = pd[pd['labels']==0]['means'].mean()
  stddev = pd['means'].std()
  pd['means'] = (pd['means'] - mean) / stddev
  print(len(pd))
  print(len(pd[pd['labels']==0]))
  return pd, mean, stddev

def memoize(fname, model):
  if exists(fname):
    with model:
      with open(fname,'rb') as buff:
        trace = pickle.load(buff)
  else:
    with model:
      trace = sample(tune = 1000, draws = 3000, return_inferencedata = True)
      with open(fname, 'wb') as buff:
        pickle.dump(trace,buff)
  return trace

def pdSplit(pd):
  splitPoint = int(0.8*(len(pd)))
  return pd[:splitPoint], pd[splitPoint:]

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
    deut = Normal('δ', mu = np.zeros(cols), sigma = 1, shape=(cols))
    ll  = at.dot(xs, beta)
    ll += at.dot(xs, deut) * ls
    error = HalfCauchy('ε', beta = 1)
    likelihood = Normal('ys', mu = ll, sigma = error, observed = ys,shape=(rows))
    trace = memoize('mixture.model', model)
  print('sampling finished')
  #
  varNames = ['ε', 'β', 'δ']
  az.plot_trace(trace,figsize=(20,20))
  plt.savefig(f'mixture-{k}-trace.pdf')
  az.plot_posterior(trace, var_names = varNames,figsize=(20,20))
  plt.savefig(f'mixture-{k}-posterior.pdf',)
  #
  s = az.summary(trace, var_names = varNames)
  print(s)
  #
  # plot on y=0 all samples with label 0, y=1, label 1; x-axis should have adjusted mu
  #
  return {} # return { 'mu': mu, 'beta1': beta1, 'deut': deut, 'err': error }

#
#
# NOTE a variant with 3-mers is possible but leads to divergence of the sampler somewhat often
# TODO normalize everything; but over all data.

def modelLogistic(pdAll, k):
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
    XS = mc.MutableData('XS', xs)
    LS = mc.MutableData('LS', ls)
    # should be at zero, since we normalized
    mu = Normal('μ',0, sigma=1)
    beta = Normal('β', mu = np.zeros(cols), sigma=10, shape=(cols))
    ll = mu + at.dot(XS, beta)
    p = mc.Deterministic('p', mc.invlogit(ll))
    likelihood = mc.Bernoulli('ys', p=p, observed = LS)
    trace = memoize('logistic.model', model)
  print('sampling finished')
  varNames = ['μ', 'β']
  az.plot_posterior(trace, var_names = varNames,figsize=(20,20))
  plt.savefig(f'logistic-{k}-posterior.pdf',)
  s = az.summary(trace, var_names = varNames)
  print(s)
  print(trace['posterior'])
  #
  print('posterior predictive')
  with model:
    mc.set_data({'XS': xs, 'LS': ls})
    test = mc.sample_posterior_predictive(trace)
    print(test)

#

def main ():
  print(f'PyMC v{mc.__version__}')
  labels={}
  labels['0'] = getIdLabels('barcode14.ids')
  labels['30'] = getIdLabels('barcode15.ids')
  labels['100'] = getIdLabels('barcode16.ids')
  dir = '/shared/choener/Workdata/heavy_atoms_deuterium_taubert/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #dir = '.'
  #dir = '/data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  pdAll = None
  if exists('./pdAll.pandas'):
    pdAll = pandas.read_pickle('./pdAll.pandas')
  else:
    pds = []
    cnt = 0
    for path in Path(dir).rglob(f'*.fast5'):
      pd = fast5Handler(labels,path)
      pds.append(pd)
      cnt += 1
      #if cnt >= 10: break
    allpds = pandas.concat(pds)
    allpds.to_pickle('./pdAll.pandas')
    pdAll = allpds
  relNucComp(pdAll)
  nucleotideHistogram(pdAll)
  # only work with data that has known labels
  pdd = pdAll[pdAll['labels'] != -1]
  pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  pd, pdmean, pdstddev = normalize(pdd)
  # TODO split off train / test
  print(len(pd))
  pd = pd.sample(frac=1)
  #modelDirichletNucs(pd)
  #modelMixtureNucs(pd[:splitPoint],k=1)
  #modelMixtureNucs(pd[:splitPoint],k=2)
  modelLogistic(pd, k=1)
  #testModel (pd[splitPoint:], mdl)

if __name__ == "__main__":
  main()

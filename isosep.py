#!/usr/bin/env python3

from collections import Counter
from itertools import chain
from pandas.plotting import scatter_matrix
from pymc3 import *
from pymc3 import Model, Normal, HalfCauchy, sample
import arviz as az
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import pymc3 as mc

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
    self.k1 = Counter(chain.from_iterable(seq))
    self.k2 = Counter(s2)

#

def maximalRunOf(fast5):
  rs = []
  for k in fast5['Analyses/'].keys():
    if 'Segmentation' in k:
      rs.append(int(k.split('_')[1]))
  return max(rs)

#

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
  # TODO construct data frame, where we can then later cut things away
  return pandas.DataFrame(
    { 'nstats': ns
    , 'means': ms
    , 'labels': ls
    })

def createData(ks,pd):
  xs = []
  for n in pd['nstats']:
    s1 = sum(n.k1.values())
    arr = np.array([n.k1[k] / s1 for k in ks])
    xs.append(arr)
  return xs

# TODO run a simple mono- and dinucleotide model for the mean of the signal. Do this for
# raw data
# z-score data
# EMA data
# beta * [A-count,C-count,G-count,T-count,1] ~ observed mean
# now, we need to add the change due to deuterium, which might be a shift in beta?

def model (pdd):
  pd=pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  print(len(pd))
  ks = set()
  for n in pd['nstats']:
    for k in n.k1.keys():
      ks.add(k)
  # create vectorized data
  xs = createData(ks,pd)
  with Model():
    XS = Data('X', value = np.vstack(xs))
    ys = Data('y', value = np.array(pd['means']))
    ls = Data('c', value = np.array(pd['labels']))
    intercept = Normal('Intercept', 0, sd=30)
    beta = Normal('β', 0, sd=30, shape=(len(ks), 1))
    # additional change in beta, if the label equals 1
    # gamma = Normal('γ', 0, sd=30, shape=(len(ks), 1))
    mu = intercept + math.dot(XS, beta) # + ls * gamma)
    epsilon = HalfCauchy('ε', 5)
    likelihood = Normal('ys', mu, epsilon, observed = ys)
    trace = sample(1000, chains=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    #scatter_matrix(traceDF, figsize=(8,8))
    traceplot(trace)
    plt.savefig('fuck.pdf')

def main ():
  print(f'PyMC3 v{mc.__version__}')
  labels={}
  labels['0'] = getIdLabels('barcode14.ids')
  labels['30'] = getIdLabels('barcode15.ids')
  labels['100'] = getIdLabels('barcode16.ids')
  dir = '/shared/choener/Workdata/heavy_atoms_deuterium_taubert/tests'
  pd = fast5Handler(labels,dir + '/' + 'FAR96927_a59606f5_0.fast5')
  model(pd)

if __name__ == "__main__":
  main()

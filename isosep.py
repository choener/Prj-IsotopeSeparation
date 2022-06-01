#!/usr/bin/env python3

from collections import Counter, OrderedDict
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
from pathlib import Path
import theano.tensor as tt

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
  print(len(ns))
  # TODO construct data frame, where we can then later cut things away
  return pandas.DataFrame(
    { 'nstats': ns
    , 'means': ms
    , 'labels': ls
    })

def createData(ks1,ks2,pd):
  xs1 = []
  xs2 = []
  for n in pd['nstats']:
    s1 = sum(n.k1.values())
    s2 = sum(n.k2.values())
    arr1 = np.array([n.k1[k] / s1 for k in ks1])
    arr2 = np.array([n.k2[k] / s2 for k in ks2])
    xs1.append(arr1)
    xs2.append(arr2)
  return np.asmatrix(xs1), np.asmatrix(xs2)

# TODO run a simple mono- and dinucleotide model for the mean of the signal. Do this for
# raw data
# z-score data
# EMA data
# beta * [A-count,C-count,G-count,T-count,1] ~ observed mean
# now, we need to add the change due to deuterium, which might be a shift in beta?

def model (pdd):
  pd=pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  print(len(pd))
  ks1 = set()
  ks2 = set()
  for n in pd['nstats']:
    for k in n.k1.keys():
      ks1.add(k)
    for k in n.k2.keys():
      ks2.add(k)
  ks1=list(ks1)
  ks1.sort()
  ks2=list(ks2)
  ks2.sort()
  xsMat1, xsMat2 = createData(ks1,ks2,pd)
  with Model():
    ys = Data('y', value = np.array(pd['means']))
    ls = Data('c', value = np.array(pd['labels']))
    #
    xs1 = Data('xs1', value = xsMat1)
    rows1, cols1 = xsMat1.shape
    baseDisp1  = Dirichlet('bs dsp 1', np.ones(cols1))
    baseScale1 = Normal('bs scl 1', 0, sd=3)
    deutDisp1  = Dirichlet('dt dsp 1', np.ones(cols1))
    deutScale1 = Normal('dt scl 1', 0, sd=3)
    mu = np.mean(pd['means'])
    mu += baseScale1 * tt.dot(xs1,baseDisp1)
    mu += deutScale1 * tt.dot(xs1,deutDisp1) * ls
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
    likelihood = Normal('ys', mu, epsilon, observed = ys)
    trace = sample(1000, chains=4, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    #scatter_matrix(traceDF, figsize=(8,8))
    traceplot(trace)
    plt.savefig('traces.pdf')

def main ():
  print(f'PyMC3 v{mc.__version__}')
  labels={}
  labels['0'] = getIdLabels('barcode14.ids')
  labels['30'] = getIdLabels('barcode15.ids')
  labels['100'] = getIdLabels('barcode16.ids')
  dir = '/shared/choener/Workdata/heavy_atoms_deuterium_taubert/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #dir = '.'
  #dir = '/data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  pds = []
  for path in Path(dir).rglob(f'*.fast5'):
    pd = fast5Handler(labels,path)
    pds.append(pd)
    #break
  pdAll = pandas.concat(pds)
  # only work with data that has known labels
  pd = pdAll[pdAll['labels'] != -1]
  model(pd)

if __name__ == "__main__":
  main()

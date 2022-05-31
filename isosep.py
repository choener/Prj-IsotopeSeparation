#!/usr/bin/env python3

import h5py
from collections import Counter
from itertools import chain
import pymc3 as mc
import numpy as np
from pymc3 import Model, Normal, HalfCauchy, sample
from pymc3 import *
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

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

def fast5Handler (fname):
  print(fname)
  fast5 = h5py.File(fname, 'r')
  ns = []
  ms = []
  for r in fast5.keys():
    s = maximalRunOf(fast5[r])
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
    break
  fast5.close()
  return (ns,ms)

# TODO run a simple mono- and dinucleotide model for the mean of the signal. Do this for
# raw data
# z-score data
# EMA data
# beta * [A-count,C-count,G-count,T-count,1] ~ observed mean
# now, we need to add the change due to deuterium, which might be a shift in beta?

def model (ns, ms):
  ks = set()
  print(ns)
  for n in ns:
    for k in n.k1.keys():
      ks.add(k)
  # create vectorized data
  xs = []
  for n in ns:
    s1 = sum(n.k1.values())
    arr = np.array([n.k1[k] / s1 for k in ks])
    xs.append(arr)
  XS = Data('X', value = np.vstack(xs))
  ys = Data('y', value = np.array(ms))
  print(XS)
  print(ys)
  with Model():
    intercept = Normal('Intercept', 0, sd=30)
    beta = Normal('beta', 0, sd=30, shape=(len(ks)))
    epsilon = HalfCauchy('epsilon', 5)
    mu = intercept + beta * XS
    likelihood = Normal('ys', mu, epsilon, observed = ys)
    trace = sample(10)
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(8,8))
    plt.savefig('fuck.png')

def main ():
  print('enter')
  dir = '/data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  ns, ms = fast5Handler(dir + '/' + 'FAR96927_a59606f5_0.fast5')
  model(ns, ms)

if __name__ == "__main__":
  main()

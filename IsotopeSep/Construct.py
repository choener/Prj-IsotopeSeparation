
# The summary statistics construct. Loops over Fast5 files, extracts the information we can use and
# constructs a pickle, for fasta re-runs (since we have literally 100's of GByte of data).

from os.path import exists
from pathlib import Path
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import sys

import Fast5
import Stats

# Infrastructure construct, which tells us the comparison / classification items.
# NOTE Currently this is a bi-classification, later on we should consider a structured 3-label
# system, since supposedly the observed strength should increase with more higher isotope
# concentration.
#
# The construct will either read the reads or extract construct information from a pickle. After
# each read has been processed, summary statistics are immediately generated, otherwise we risk
# memory overflows, since intermediate data is huge.

class Construct:
  # Initialize (and pickle)
  def __init__(self, barcodes, reads, pickleDir, limitReads = None, plotSquiggle = None):
    self.labels = {}
    self.pickleDir = ""
    self.reads = reads
    self.summaryStats = None        # data frame containing all summary statistics
    assert len(barcodes)==2
    assert len(reads)>0
    for pcnt,barcode in barcodes:
      log.info(f'{int(pcnt):3d}%  ->  barcode file: {barcode:s}')
      self.labels[pcnt] = getIdLabels(barcode)
    # Check that labels are unique, performs intersection test on all pairs of labels
    log.info('label check')
    for l in self.labels.keys():
      for k in self.labels.keys():
        if l>=k: continue # check upper triangular matrix
        s = self.labels[l].intersection(self.labels[k])
        if len(s) > 0:
          sys.exit(f'{len(s)} non-unique labels for the label keys {l} and {k}, exiting now')
    # either load load from pickle dir (and save if first time running the data), or just go through
    # all data
    if pickleDir is not None:
      self.pickleDir = pickleDir
      self.pickleOrRead(limitReads, plotSquiggle)

  # Extract summary stats from pickle or read from actual reads
  def pickleOrRead(self, limitReads = None, plotSquiggle = None):
    fname = self.pickleDir + '/summaryStats.pandas'
    if exists(fname):
      self.summaryStats = pandas.read_pickle(fname)
    else:
      cnt = 0
      totReads = 0
      accum = SummaryStats(labels = self.labels)
      for path in self.reads:
        log.info(f'READ PATH: {path}')
        for rname in Path(path).rglob(f'*.fast5'):
          cnt += 1
          if limitReads is not None and int(limitReads) < cnt:
            break
          maxCnt = None
          if limitReads is not None:
            maxCnt = int(limitReads) - totReads
          accum, rcnt = Fast5.fast5Handler(rname,accum, maxCnt)
          totReads += rcnt
      accum.postFile()



# The summary statistics we use for our models.
# TODO switch to pandas frame here? However, pandas.append will have to recreate data frames all the
# time ...

class SummaryStats (Fast5.Fast5Accumulator):
  def __init__ (self, labels = None):
    self.labelLookup = labels
    self.readIDs = []
    self.label = []
    self.preMedian = []
    self.sufMedian = []
    self.sufMad = []
    self.numNucleotides = []
    self.k1Medians = []
    self.k1Mads = []
    self.k3Medians = []
    self.k3Mads = []
    self.k5Medians = []
    self.k5Mads = []
  def insRead (self, preRaw, segmented, nucs, rid):
    # TODO consider using limitReads to limit us to this limit but for each label type
    self.readIDs.append(rid)
    self.label.append(labelFromRId(self.labelLookup,rid))
    self.preMedian.append(np.median(preRaw))
    suf = np.concatenate(segmented)
    self.sufMedian.append(np.median(suf))
    self.sufMad.append(Stats.medianAbsoluteDeviation(suf))
    self.numNucleotides.append(len(nucs))
    kmer1Med,kmer1Mad = kmerMedians(1,nucs,segmented)
    kmer3Med,kmer3Mad = kmerMedians(3,nucs,segmented)
    kmer5Med,kmer5Mad = kmerMedians(5,nucs,segmented)
    self.k1Medians.append(kmer1Med)
    self.k1Mads.append(kmer1Mad)
    self.k3Medians.append(kmer3Med)
    self.k3Mads.append(kmer3Mad)
    self.k5Medians.append(kmer5Med)
    self.k5Mads.append(kmer5Mad)
  def fixup (self):
    # TODO fix up the "0" entries in *Med, *Mad using the average of all other entries
    # TODO create pandas dataframe
    pass
  # Draw plots for summary statistics
  def postFile (self):
    pre = np.array(self.preMedian)
    df = pd.DataFrame(data = { 'rid': np.array(self.readIDs)
                             , 'med': np.array(self.sufMedian)
                             , 'medX': (np.array(self.sufMedian) - pre + np.median(pre))
                             , 'label': np.array(self.label)
                             })
    sb.violinplot(data=df, x='label', y='medX')
    plt.savefig('postfile.pdf', bbox_inches='tight')
    plt.close()
    self.kXmedians('k1.pdf', self.k1Medians)
    self.kXmedians('k3.pdf', self.k3Medians)
    self.kXmedians('k5.pdf', self.k5Medians)
    self.kXmedians('k1mad.pdf', self.k1Mads)
    self.kXmedians('k3mad.pdf', self.k3Mads)
    self.kXmedians('k5mad.pdf', self.k5Mads)
    plt.close()
  # TODO produce random subset, if too many sufTy ...
  def kXmedians(self,outname,kwhat):
    assert len(kwhat) > 0
    xsize = max(4,min(1 * len(kwhat[0]),2**8))
    df = pd.DataFrame(data = { 'label': np.concatenate([ np.repeat(x, len(y)) for x,y in zip(self.label, kwhat) ])
                             , 'pA': fixValuesToMedian(np.concatenate([ np.array(x) for x in kwhat ]))
                             , 'sufTy': np.concatenate([np.array(range(0,len(x))) for x in kwhat])
                             })
    df = df.dropna()
    uniqLen = len(np.unique(df['label']))
    splt=False
    if uniqLen == 2: splt = True
    plt.figure(figsize=(xsize,8))
    sb.violinplot(data=df, x='sufTy', y='pA', hue='label', split=splt, cut=0)
    plt.savefig(outname, bbox_inches='tight')
    plt.close()
    # random subset
    uniqueSufTys = np.random.permutation(np.unique(df['sufTy']))
    if len(uniqueSufTys) > 65:
      subset = uniqueSufTys[0:63]
      subdf = df[df['sufTy'].isin(subset)]
      plt.figure(figsize=(64,8))
      sb.violinplot(data=subdf, x='sufTy', y='pA', hue='label', split=splt, cut=0)
      plt.savefig("sub_" + outname, bbox_inches='tight')
      plt.close()

# replace values far away from the median by the median without these values
def fixValuesToMedian(xs):
  med = np.median(xs)
  mad = Stats.medianAbsoluteDeviation(xs)
  chk = med - 2*mad
  xs[xs < chk] = med
  return xs

def labelFromRId(labels, rid):
  for k,v in labels.items():
    if rid in v:
      return k
  return None

# Take the full information for reads and generates summary statistics
# label information provides a lookup ReadID -> label in [0,1], however the label might well be
# "0"/"100" and will only be "classified" in the actual model.
# xs contains { preSignals, mainSignals, nucStrings, readIDs }

# Jannes has extracted the IDs which correspond to 0%, 30%, 100% deuterium. However, for now we only
# care about 0% vs 100% deuterium

def getIdLabels(fname):
  labels=set()
  with open(fname, 'r') as r:
    for line in r:
      labels.add(line.strip())
  return labels

# kmer string to int. Note that this hash is language-ignorant. Strings of different length yield
# the same hash, ie @kmer2int('') == kmer2int('AAA')@. One should use this RNA hash only for strings
# of the same length.
#
# This is exactly where we use this, to accumulate in vectors of the same length.

def kmer2int(kmer):
  k = 0
  lkup = { 'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3 }
  for i,c in enumerate(reversed(kmer)):
    z = lkup.get(c.upper())
    if k is not None and z is not None:
      k += z * 4**i
    if z is None:
      k = None
  return k

# int to kmer, given length of kmer

def int2kmer(k, i):
  # TODO provide inverse of kmer2int
  pass

# construct the k-mer centered around index i using the nucs sequence.
# Clamps kmers at the ends to repeated nucleotides.
# Assumes that the kmer is center-indexed.

def kmerAt(k,nucs,i):
  lr = int((k-1)/2)
  kmer=[]
  for i in range(i-lr-1,i+lr):
    j = min(max(0,i),len(nucs)-1)
    n = nucs[j]
    kmer.append(n)
  return ''.join(kmer)

# Return median information for each possible kmer, unseen kmers get 0 and we should impute the
# missing information later!

def kmerMedians (kmerLen, nucs, segments):
  assert len(nucs) == len(segments)
  medians = {}
  for i,s in enumerate(segments):
    k = kmerAt(kmerLen,nucs,i)
    m = np.median(s)
    vs = medians.get(k,[])
    vs.append(m)
    medians[k] = vs
  medianVec = np.zeros(4**kmerLen)
  madVec = np.zeros(4**kmerLen)
  for k,v in medians.items():
    medianVec[kmer2int(k)] = np.mean(v)
    madVec[kmer2int(k)] = Stats.medianAbsoluteDeviation(v)
  return medianVec, madVec


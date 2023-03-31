
# The summary statistics construct. Loops over Fast5 files, extracts the information we can use and
# constructs a pickle, for fasta re-runs (since we have literally 100's of GByte of data).

# TODO test if strings are interned. IE. "c is d" returns true for "equal" strings. If not, there is
# sys.intern(str) which will internalize the string.

from os.path import join
from scipy import stats
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
import scipy

import Fast5
import Stats
import Kmer

# Infrastructure construct, which tells us the comparison / classification items.
# NOTE Currently this is a bi-classification, later on we should consider a structured 3-label
# system, since supposedly the observed strength should increase with more higher isotope
# concentration.
#
# The construct will either read the reads or extract construct information from a pickle. After
# each read has been processed, summary statistics are immediately generated, otherwise we risk
# memory overflows, since intermediate data is huge.

# TODO parallelization of extracting data from reads

class Construct:

  # Initialize (and pickle)
  def __init__(self):
    self.labels = {}
    self.finishedReads = set()      # store which reads we have finished working with
    self.dfgroups = []
    self.summaryStats = None        # data frame containing all summary statistics

  def addbarcode(self, pcnt, barcode):
    log.info(f'barcode file: {barcode:s}  ->  {int(pcnt):3d}%')
    for lbl in getIdLabels(barcode):
      self.labels[lbl] = float(pcnt) / 100
    #self.labels[pcnt] = self.labels.get(pcnt,set()).union(getIdLabels(barcode))
    #log.info('label check')
    #for l in self.labels.keys():
    #  for k in self.labels.keys():
    #    if l>=k: continue # check upper triangular matrix
    #    s = self.labels[l].intersection(self.labels[k])
    #    if len(s) > 0:
    #      sys.exit(f'{len(s)} non-unique labels for the label keys {l} and {k}, exiting now')

  def __len__(self):
    return (len(self.finishedReads))

  # Add a dataframe with kmer information for further processing
  # this will reduce the dataframe to key pairs (kmer,relative d2o) with mean values for medianZ and
  # madZ.
  # TODO add column that contains the normalized (prefix information). Needs to be scaled the same
  # way "madZ" is scaled. That information is available in reads.csv.zst
  def addkmerdf(self,kmer,df, rds):
    rds['pfxZ'] = (rds['pfxMedian'] - rds['median']) / rds['mad']
    df = df.merge(rds[['read','pfxZ']], on='read')
    # restrict processing to rows with correct kmer length
    k = df['k']
    k = k.apply(lambda x: len(x) == int(kmer))
    df = df[k]
    # transform mad via boxcox: won't always work, if not enough data available
    #print(df[df['madZ']==0])
    #mads, lmbd = scipy.stats.boxcox(df['madZ'])
    #df = df.assign(madZbc = mads)
    #
    # now determine relative d2o content based on read and known labels
    rs = df['read']
    rs = rs.apply(lambda x: self.labels.get(x[5:],np.nan))
    df = df.assign(rel = rs)
    # remove all reads, where we have no deuterium concentration assignment
    df = df.dropna()
    # somewhat involved construction of missing kmers, set to zero!
    missing = []
    for r,g in df.groupby('read'):
      # construct missing indices
      mixs = set(Kmer.gen(int(kmer)))
      mixs = mixs.difference(set(g['k']))
      mixs = pd.MultiIndex.from_product([[r],mixs], names=['read','k'])
      mdf = pd.DataFrame(index=mixs, columns = df.columns.drop(['read','k'])).fillna(0)
      mdf = mdf.assign(rel = g['rel'].mean())
      missing.append(mdf)
    # prepare frame
    df = df.set_index(['read','k'])
    df = pd.concat([df] + missing)
    df = df.sort_index()
    self.dfgroups.append(df)

  def mergegroups(self):
    sing = pd.concat(self.dfgroups)
    self.dfgroups = [sing]

  # save a Construct to file
  def save(self, fname):
    with open(fname, 'wb') as f:
      pickle.dump(self.__dict__,f)

  # construct = Construct.load("f.name")
  @classmethod
  def load(cls, fname):
    c = cls(barcodes = [])
    with open(fname, 'rb') as f:
      dict = pickle.load(f)
      c.__dict__.update(dict)
      assert (c.summaryStats is not None)
      sz = len(c.finishedReads)
      assert (sz == len(c.summaryStats))
      return c

  # will merge the "other" datastructure into ourself. Doesn't do any fancy error checking.
  #def merge(self, other):
  #  self.labels.update(other.labels)
  #  self.finishedReads.update(other.finishedReads)
  #  if self.summaryStats is None:
  #    self.summaryStats = other.summaryStats
  #  else:
  #    self.summaryStats.merge(other.summaryStats)

  # handle all reads within a file. If maxCnt is given, then stop after this many new reads have
  # been collected. This allows for stopping and restarting.
  #
  # TODO split the loop into two, parallelizing the accum.insRead?
  #
  def handleReadFile(self, fname, limitNewReadCnt = None):
    cnt = 1
    if self.summaryStats is None:
      self.summaryStats = SummaryStats()
    rids = Fast5.fast5Reads(fname)
    assert (rids is not None)
    rids = set(rids)
    rids.difference_update(self.finishedReads)
    if limitNewReadCnt is None:
      limitNewReadCnt = len(rids)
    numKeys = min(len(rids), limitNewReadCnt)
    # only new reads considered
    for r in rids:
      if cnt > limitNewReadCnt:
        break
      preRaw, segmented, nucs = Fast5.fast5ReadData(fname, r, i = cnt, numKeys = numKeys)
      self.summaryStats.insRead(self.labels, preRaw, segmented, nucs, r)
      self.finishedReads.add(r)
      cnt = cnt + 1
    return cnt



# The summary statistics we use for our models.
# TODO switch to pandas frame here? However, pandas.append will have to recreate data frames all the
# time ...

class SummaryStats (Fast5.Fast5Accumulator):
  def __init__ (self):
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
    self.k1LenMean = []
    self.k1LenVar  = []
    self.k3LenMean = []
    self.k3LenVar  = []
    self.k5LenMean = []
    self.k5LenVar  = []
    self.uniqueSufTys = None

  # the length method only returns the length of one of its members, since it is assumed that all
  # members have the same length
  def __len__(self):
    assert(len(self.readIDs) == len(self.k5LenVar))
    return (len(self.readIDs))

  def insRead (self, labels, preRaw, segmented, nucs, rid):
    # TODO consider using limitReads to limit us to this limit but for each label type
    self.readIDs.append(rid)
    self.label.append(labelFromRId(labels,rid))
    self.preMedian.append(np.median(preRaw))
    suf = np.concatenate(segmented)
    self.sufMedian.append(np.median(suf))
    self.sufMad.append(Stats.medianAbsoluteDeviation(suf))
    self.numNucleotides.append(len(nucs))
    kmer1Med,kmer1Mad,kmer1LenMean,kmer1LenVar = kmerStatistics(1,nucs,segmented)
    kmer3Med,kmer3Mad,kmer3LenMean,kmer3LenVar = kmerStatistics(3,nucs,segmented)
    kmer5Med,kmer5Mad,kmer5LenMean,kmer5LenVar = kmerStatistics(5,nucs,segmented)
    self.k1Medians.append(kmer1Med)
    self.k1Mads.append(kmer1Mad)
    self.k3Medians.append(kmer3Med)
    self.k3Mads.append(kmer3Mad)
    self.k5Medians.append(kmer5Med)
    self.k5Mads.append(kmer5Mad)
    self.k1LenMean.append(kmer1LenMean)
    self.k1LenVar.append(kmer1LenVar)
    self.k3LenMean.append(kmer3LenMean)
    self.k3LenVar.append(kmer3LenVar)
    self.k5LenMean.append(kmer5LenMean)
    self.k5LenVar.append(kmer5LenVar)

  def merge(self, other):
    self.readIDs.extend(other.readIDs)
    self.label.extend(other.label)
    self.preMedian.extend(other.preMedian)
    self.sufMedian.extend(other.sufMedian)
    self.sufMad.extend(other.sufMad)
    self.numNucleotides.extend(other.numNucleotides)
    self.k1Medians.extend(other.k1Medians)
    self.k1Mads.extend(other.k1Mads)
    self.k3Medians.extend(other.k3Medians)
    self.k3Mads.extend(other.k3Mads)
    self.k5Medians.extend(other.k5Medians)
    self.k5Mads.extend(other.k5Mads)
    self.k1LenMean.extend(other.k1LenMean)
    self.k1LenVar.extend(other.k1LenVar)
    self.k3LenMean.extend(other.k3LenMean)
    self.k3LenVar.extend(other.k3LenVar)
    self.k5LenMean.extend(other.k5LenMean)
    self.k5LenVar.extend(other.k5LenVar)

  def fixup (self):
    # TODO fix up the "0" entries in *Med, *Mad using the average of all other entries
    # TODO create pandas dataframe
    pass

  # Draw plots for summary statistics
  def postFile (self, odir):
    # temporary fixups (do not modify original data inplace)
    pre = np.array(self.preMedian)
    pre[np.where(np.isnan(pre))]=np.nanmedian(pre)
    df = pd.DataFrame(data = { 'rid': np.array(self.readIDs)
                             , 'med': np.array(self.sufMedian)
                             , 'medX': (np.array(self.sufMedian) - pre + np.median(pre))
                             , 'label': np.array(self.label)
                             })
    print(df['label'])
    sb.violinplot(data=df, x='label', y='medX')
    plt.savefig(join(odir,'postfile.pdf'), bbox_inches='tight')
    plt.close()
    self.uniqueSufTys = None
    self.kXmedians(odir,'k1mad.pdf', self.k1Mads, y='pA +- mad')
    self.kXmedians(odir,'k1.pdf', self.k1Medians)
    self.uniqueSufTys = None
    self.kXmedians(odir,'k3mad.pdf', self.k3Mads, y='pA +- mad')
    self.kXmedians(odir,'k3.pdf', self.k3Medians)
    self.uniqueSufTys = None
    self.kXmedians(odir,'k5lenmedian.pdf', self.k5LenMean, y='length mean')
    self.kXmedians(odir,'k5lenmad.pdf', self.k5LenVar, y='length var')
    self.kXmedians(odir,'k5mad.pdf', self.k5Mads, y='pA +- mad')
    self.kXmedians(odir,'k5.pdf', self.k5Medians)
    plt.close()

  # TODO produce random subset, if too many sufTy ...
  def kXmedians(self,odir,oname,kwhat, y=None):
    lbl = 'pA'
    if y is not None:
      lbl = y
    assert len(kwhat) > 0
    xsize = max(4,min(1 * len(kwhat[0]),2**8))
    df = pd.DataFrame(data = { 'label': np.concatenate([ np.repeat(x, len(y)) for x,y in zip(self.label, kwhat) ])
                             , lbl: fixValuesToMedian(np.concatenate([ np.array(x) for x in kwhat ]))
                             , 'sufTy': np.concatenate([np.array(range(0,len(x))) for x in kwhat])
                             })
    df = df.dropna()
    df = df[(np.abs(stats.zscore(df[lbl]))<3)]
    assert df is not None
    uniqLen = len(np.unique(df['label']))
    splt=False
    if uniqLen == 2: splt = True
    plt.figure(figsize=(xsize,8))
    sb.violinplot(data=df, x='sufTy', y=lbl, hue='label', split=splt, cut=0)
    plt.savefig(join(odir,oname), bbox_inches='tight')
    plt.close()
    # random subset
    if self.uniqueSufTys is None:
      self.uniqueSufTys = np.random.permutation(np.unique(df['sufTy']))
    if len(self.uniqueSufTys) > 65:
      subset = self.uniqueSufTys[0:63]
      subdf = df[df['sufTy'].isin(subset)]
      plt.figure(figsize=(64,8))
      sb.violinplot(data=subdf, x='sufTy', y=lbl, hue='label', split=splt, cut=0)
      plt.savefig(join(odir,"sub_" + oname), bbox_inches='tight')
      plt.close()

# replace values far away from the median by the median without these values
def fixValuesToMedian(xs):
  med = np.median(xs)
  mad = Stats.medianAbsoluteDeviation(xs)
  chk = med - 2*mad
  xs[xs < chk] = med
  return xs

# lookup the label from the RId; note the potential need for a prefix!
def labelFromRId(labels, ridd):
  rid = ridd.removeprefix('read_')
  for k,v in labels.items():
    if rid in v:
      return (float(k) / 100)
  return -1

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
#
# TODO maybe return a class object

def kmerStatistics (kmerLen, nucs, segments):
  assert len(nucs) == len(segments)
  medians = {}
  lengths = {}
  for i,s in enumerate(segments):
    k = kmerAt(kmerLen,nucs,i)
    m = np.median(s)
    vs = medians.get(k,[])
    vs.append(m)
    medians[k] = vs
    ls = lengths.get(k,[])
    ls.append(len(s))
    lengths[k] = ls
  medianVec = np.zeros(4**kmerLen)
  madVec = np.zeros(4**kmerLen)
  lengthMedianVec = np.zeros(4**kmerLen)
  lengthMadVec = np.zeros(4**kmerLen)
  #log.info(f'kmerStatistics.medians {kmerLen}')
  for k,v in medians.items():
    medianVec[kmer2int(k)] = np.mean(v)
    madVec[kmer2int(k)] = Stats.medianAbsoluteDeviation(v)
  #log.info(f'kmerStatistics.length {kmerLen}')
  for k,v in lengths.items():
    lengthMedianVec[kmer2int(k)] = np.median(v)
    lengthMadVec[kmer2int(k)]  = Stats.medianAbsoluteDeviation(v) # TODO rename tgt
  #log.info(f'kmerStatistics DONE')
  return medianVec, madVec, lengthMedianVec, lengthMadVec


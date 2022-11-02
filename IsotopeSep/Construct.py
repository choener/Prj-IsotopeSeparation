
# The summary statistics construct. Loops over Fast5 files, extracts the information we can use and
# constructs a pickle, for fasta re-runs (since we have literally 100's of GByte of data).

from os.path import exists
from pathlib import Path
import logging as log
import numpy as np
import pandas
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
    #relNucComp(self.summaryStats)
    #nucleotideHistogram(self.summaryStats)

  # Extract summary stats from pickle or read from actual reads
  def pickleOrRead(self, limitReads = None, plotSquiggle = None):
    fname = self.pickleDir + '/summaryStats.pandas'
    if exists(fname):
      self.summaryStats = pandas.read_pickle(fname)
    else:
      pds = []
      cnt = 0
      accum = Fast5.AccumDF()
      for path in self.reads:
        log.info(f'READ PATH: {path}')
        for rname in Path(path).rglob(f'*.fast5'):
          cnt += 1
          if limitReads is not None and int(limitReads) < cnt:
            break
          # contains data for approx. 4000 reads or so
          accum = Fast5.fast5Handler(rname,accum)
          # calculate summary statistics to get 4000 rows of information, with huge number of
          # columns
          #summarised = genSummaryStats(self.labels, fdata)
          #pds.append(summarised)
      #self.summaryStats = pandas.concat(pds)
      #self.summaryStats.to_pickle(fname)



# 

class SummaryStats (Fast5.Fast5Accumulator):
  def __init__ (self):
    self.readIDs = []
    self.preMedian = []
    self.sufMedian = []
    self.sufMad = []
    self.numNucleotides = []
    pass
  def insRead (self, preRaw, segmented, nucs, rid):
    self.readIDs.append(rid)
    self.preMedian.append(np.median(preRaw))
    suf = np.concatenate(segmented)
    self.sufMedian.append(np.median(suf))
    self.sufMad.append(Stats.medianAbsoluteDeviation(suf))
    self.numNucleotides.append(len(nucs))
    pass
  # will convert the internal structures to arrays, where beneficial
  #def convertToArrays(self):
  #  self.sufMad = np.array(self.sufMad)
  #  pass

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
  pass


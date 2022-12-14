
# Fast5 handling methods

import abc
import h5py
import logging as l
import pandas as pandas
import sys



# TODO

def maximalRunOf(fast5):
  rs = []
  for k in fast5['Analyses/'].keys():
    if 'Segmentation' in k:
      rs.append(int(k.split('_')[1]))
  return max(rs)

# Extract the raw signal from a fast5 file at the given key

def rawSignal(fast5, key):
  return fast5[key]['Raw/Signal'][:]

# Takes a raw signal and returns a signal in pico-ampere

def undigitise(fast5, key, raw):
  ch = fast5[key][f'channel_id']
  digitisation = ch.attrs['digitisation']
  offset = ch.attrs['offset']
  range = ch.attrs['range']
  scale = range / digitisation
  pA = scale * (raw + offset)
  return pA

# Split the beginning part of the raw signal, so that we have (prefix,actual signal)

def splitRawSignal(fast5, key, rs):
  segment = maximalRunOf(fast5[key])
  start = fast5[key][f'Analyses/Segmentation_{segment:03d}/Summary/segmentation'].attrs['first_sample_template']
  return rs[:start], rs[start:]

# Extract the move table, which tells us when ONT software thinks new nucleotide information begins

def moveTable(fast5, key):
  segment = maximalRunOf(fast5[key])
  return list(fast5[key][f'Analyses/Basecall_1D_{segment:03d}/BaseCalled_template/Move'][()])

# Take the raw signal (with everything before 'first_sample_template' already removed), and separate
# according to the move table. This will yield another vector with the same length as the move table
# and also the same length as the base called nucleotides. This vector will contain, in each
# element, the raw signals corresponding to the nucleotide being base called.
#
# The move table stores moves in a stride of 5, hence 5 * length_move_table is approx. the length of
# the raw signal vector (without first_sample_template).

def segmentSignal (rawSignal, moveTable, stride=5):
  i = iter(rawSignal)
  s = len(moveTable)
  numZeros = max (0, int(len(rawSignal)/5 - s))
  sizes = moveTableToLengths(moveTable+([0] * numZeros),stride)
  return [[next(i) for _ in range(sz)] for sz in sizes]

# Given a moveTable [1,1,0,0,1,...] create the sum(moveTable) length table of lengths of moves.

def moveTableToLengths (moveTable, stride=5):
  ls = []
  # go through each move
  for m in moveTable:
    # new event signaled by "1"
    if m > 0:
      ls.append(stride)
    # same event, increase last element by stride steps, but only if there has been at least one
    # event
    elif (len(ls)>0):
      ls[-1] += stride
  return ls

# Extract the base called sequence from a fast5 file at the given key

def nucleotides(fast5, key):
  segment = maximalRunOf(fast5[key])
  return fast5[key][f'Analyses/Basecall_1D_{segment:03d}/BaseCalled_template/Fastq'][()].decode('utf-8').split('\n')[1]

# The fast5Handler iterates over each fast5 file, extracts the raw signal and transforms the signal
# to pA (pico Ampere). Returned are presignals, main signals (split again for each move point), the
# nucleotide strings and the ID for each read.

def fast5Handler (fname, accumulator, maxCnt = None):
  fast5 = h5py.File(fname, 'r')
  numKeys = len(fast5.keys())
  i=0
  for r in fast5.keys():
    # early break is possible
    if maxCnt is not None and i>=maxCnt:
      break
    i+=1
    raw = undigitise(fast5,r, rawSignal(fast5,r))
    (preRaw,sufRaw) = splitRawSignal(fast5,r,raw)
    segmented = segmentSignal(sufRaw,moveTable(fast5,r))
    nucs = nucleotides(fast5,r)
    rid = r.split('read_')[1]
    l.info(f' {i:4d}/{numKeys:4d} RID: {rid} preS: {len(preRaw):5d} sufRaw: {len(sufRaw):7d} nucs: {len(nucs):7d}   s/n: {len(sufRaw)/len(nucs):5.1f}')
    accumulator.insRead(preRaw, segmented, nucs, rid)
  fast5.close()
  return accumulator, i

# Extract all keys from a fast5 file

def fast5Reads(fname):
  fast5 = h5py.File(fname, 'r')
  ks = list(fast5.keys())
  fast5.close()
  return ks

def fast5ReadData(fname, r, i = 0, numKeys = 0):
  fast5 = h5py.File(fname, 'r')
  raw = undigitise(fast5,r, rawSignal(fast5,r))
  (preRaw,sufRaw) = splitRawSignal(fast5,r,raw)
  segmented = segmentSignal(sufRaw,moveTable(fast5,r))
  nucs = nucleotides(fast5,r)
  rid = r.split('read_')[1]
  l.info(f' {i:4d}/{numKeys:4d} RID: {rid} preS: {len(preRaw):5d} sufRaw: {len(sufRaw):7d} nucs: {len(nucs):7d}   s/n: {len(sufRaw)/len(nucs):5.1f}')
  fast5.close()
  return preRaw, segmented, nucs


# Generic accumulators for statistics

class Fast5Accumulator(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def __init__ (self):
    pass
  @abc.abstractmethod
  def __len__ (self):
    pass
  @abc.abstractmethod
  def insRead (self, preRaw, segmented, nucs, rid):
    pass
  # allows us to do any cleanup / stats that should be done after a single file has been fully read.
  @abc.abstractmethod
  def postFile (self):
    pass
  @abc.abstractmethod
  def merge(self, other):
    sys.exit("ERROR: merge has not been implemented")
    pass



# This wrapper will just accumulate data for a pandas dataframe and perform no summary statistics
# calculation

class AccumDF (Fast5Accumulator):
  def __init__ (self):
    self.preSignals = []
    self.mainSignals = []
    self.nucStrings = []
    self.readIDs = []
    pass

  # insert a single read
  def insRead (self, preRaw, segmented, nucs, rid):
    self.preSignals.append(preRaw)
    self.mainSignals.append(segmented)
    self.nucStrings.append(nucs)
    self.readIDs.append(rid)

  # merge two accumulators into one
  def mergeAccumulator(self, other):
    self.preSignals.extend(other.preSignals)
    self.mainSignals.extend(other.preSignals)
    self.nucStrings.extend(other.nucStrings)
    self.readIDs.extend(other.readIDs)

  def getDF (self):
    return pandas.DataFrame(
      { 'preSignals': self.preSignals
      , 'mainSignals': self.mainSignals
      , 'nucStrings': self.nucStrings
      , 'readIDs': self.readIDs
      })



# An accumulator that does nothing

class VoidDF (Fast5Accumulator):
  def insRead (self, __preRaw__, __segmented__, __nucs__, __rid__):
    pass


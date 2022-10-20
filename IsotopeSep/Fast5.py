
# Fast5 handling methods

from collections import Counter
from itertools import chain
import h5py
import numpy as np
import pandas as pandas
import statistics



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

# Statistics for movetabled nucleotide positions in the raw signal

def signalStatistics (splitsignal):
  stats = []
  for ns in splitsignal:
    stats.append([np.mean(ns),statistics.variance(ns),np.median(ns)])
  return np.transpose(np.array(stats))

# Extract the base called sequence from a fast5 file at the given key

def nucleotides(fast5, key):
  segment = maximalRunOf(fast5[key])
  return fast5[key][f'Analyses/Basecall_1D_{segment:03d}/BaseCalled_template/Fastq'][()].decode('utf-8').split('\n')[1]

# The fast5Handler iterates over each fast5 file, extracts the raw signal and transforms the signal
# to pA (pico Ampere). Returned are presignals, main signals (split again for each move point), the
# nucleotide strings and the ID for each read.

def fast5Handler (fname):
  fast5 = h5py.File(fname, 'r')
  preSignals = []
  mainSignals = []
  nucStrings = []
  readIDs = []
  for r in fast5.keys():
    print(r)
    raw = undigitise(fast5,r, rawSignal(fast5,r))
    (preRaw,sufRaw) = splitRawSignal(fast5,r,raw)
    segmented = segmentSignal(sufRaw,moveTable(fast5,r))
    preSignals.append(preRaw)
    mainSignals.append(segmented)
    nucStrings.append(nucleotides(fast5,r))
    readIDs.append(r.split('read_')[1])
  fast5.close()
  return pandas.DataFrame(
    { 'preSignals': preSignals
    , 'mainSignals': mainSignals
    , 'nucStrings': nucStrings
    , 'readIDs': readIDs
    })

# Store summary statistics for a time series (of a squiggle plot)

class SeriesStats:
  def __init__(self, xs):
    self.mean = statistics.mean(xs)
    self.var = statistics.variance(xs)
    self.median = statistics.median(xs)
    maddiff = xs-np.median(xs)
    self.mad = np.median(abs(maddiff))
    up = maddiff[maddiff>0]
    down = maddiff[maddiff<0]
    self.upmad = np.median(up)
    self.downmad = np.median(down)

# Store summary statistics for the nucleotides

class NucleotideStats:
  def __init__(self, seq):
    s2 = [''.join(pair) for pair in zip(seq[:-1],seq[1:])]
    s3 = [''.join(triplet) for triplet in zip(seq[:-2],seq[1:-1],seq[2:])]
    self.k1 = Counter(chain.from_iterable(seq))
    self.k2 = Counter(s2)
    self.k3 = Counter(s3)


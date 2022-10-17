
# Fast5 handling methods

from collections import Counter
from itertools import chain
import h5py
import numpy as np
import pandas as pandas
import statistics

import Plot



# TODO

def maximalRunOf(fast5):
  rs = []
  for k in fast5['Analyses/'].keys():
    if 'Segmentation' in k:
      rs.append(int(k.split('_')[1]))
  return max(rs)


# The fast5Handler iterates over each fast5 file, extracts the raw signal and transforms the signal
# to pA (pico Ampere).
#
# This is followed by transformation operations, which currently amount to a mean() operation over
# the pA of each read.

# TODO Make this generic over the k-mers

def fast5Handler (labels,fname, plotSquiggle = False):
  fast5 = h5py.File(fname, 'r')
  ns = []
  nucStats = []
  readLabels = []
  preStats = []
  i=0
  for r in fast5.keys():
    i+=1
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
    pAFull = scale * (rawSignal + offset)
    pA = pAFull[start:]
    if plotSquiggle is not None:
      Plot.plotSquiggle(plotSquiggle, 'raw-' + r, pAFull, start)
      Plot.plotSquiggle(plotSquiggle, 'hdr-' + r, pAFull[0:int(start+50)], start)
    nstats = NucleotideStats(fastq)
    ns.append(nstats)
    nucStats.append(SeriesStats(pA))
    preStats.append(SeriesStats(pAFull[0:int(max(0,start-1))]))
    k = ''
    for l in labels.keys():
      if rid in labels[l]:
        k = l
        break
    readLabels.append(k)
  fast5.close()
  # TODO construct data frame, where we can then later cut things away
  return pandas.DataFrame(
          # computated by NucleotideStats() nucleotide statistics, the relative abundance of each
          # nucletodie, 2mer, 3mer, ... as
    { 'nstats': ns
          # summary statistics of nucleotide signal
    , 'nucStats': nucStats
          # label given to this read, i.e. if deuterium present or not
    , 'labels': readLabels
          # summary statistics of pre-nucleotide (adapter? etc) signal
    , 'preStats': preStats
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

# Take the raw signal (with everything before 'first_sample_template' already removed), and separate
# according to the move table. This will yield another vector with the same length as the move table
# and also the same length as the base called nucleotides. This vector will contain, in each
# element, the raw signals corresponding to the nucleotide being base called.
#
# The move table stores moves in a stride of 5, hence 5 * length_move_table is approx. the length of
# the raw signal vector (without first_sample_template).

def segmentRawSignal (rawSignal, moveTable, stride=5):
  i = iter(rawSignal)
  sizes = moveTableToLengths(moveTable,stride)
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


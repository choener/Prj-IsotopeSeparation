
from collections import Counter
from itertools import chain
import numpy as np
import statistics

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

# TODO create classes that take the Fast5 inputs read-wise, this way we can give the fast5 function
# the functionality to turn the "big" data into summary statistics.

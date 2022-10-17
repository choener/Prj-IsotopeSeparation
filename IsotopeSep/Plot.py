
# Plotting functionality

import os
import matplotlib as pl
import matplotlib.pyplot as plt
import statistics

# Plot a (simple) time series plot for ONT data. The "firstSample" would be the first index, where
# actual data from the nucleotides is, while beforehand, we have adapters, etc.

def plotSquiggle(tgtDir, fname, xs, firstSample = None):
  fn = os.path.join(tgtDir, fname + '.pdf')
  plt.figure(figsize=(20,4), dpi=300)
  plt.plot(xs)
  if firstSample is not None:
    # position of first sample
    plt.axvline(firstSample, color = 'green')
    mdn = statistics.median(xs[0:int(max(0,firstSample-1))])
    plt.plot((0,firstSample), (mdn,mdn), color = 'red')
    # median of adapter, etc signal
  plt.savefig(fn, bbox_inches='tight')
  plt.close()
  pass


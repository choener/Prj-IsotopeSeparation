
# The summary statistics construct. Loops over Fast5 files, extracts the information we can use and
# constructs a pickle, for fasta re-runs (since we have literally 100's of GByte of data).

import sys
from os.path import exists
import pandas
from pathlib import Path

import Fast5

# Infrastructure construct, which tells us the comparison / classification items.
# NOTE Currently this is a bi-classification, later on we should consider a structured 3-label
# system, since supposedly the observed strength should increase with more higher isotope
# concentration.

class Construct:
  # Initialize (and pickle)
  def __init__(self, barcodes, reads, pickleDir, limitReads = None, plotSquiggle = None):
    self.labels = {}
    self.pickleDir = ""
    self.readSummaryStats = None
    self.reads = reads
    assert len(barcodes)==2
    assert len(reads)>0
    for b in barcodes:
      pcnt,barcode = b
      print (pcnt, barcode)
      self.labels[pcnt] = getIdLabels(barcode)
    # Check that labels are unique
    print('label check')
    for l in self.labels.keys():
      for k in self.labels.keys():
        if l>=k: continue # check upper triangular matrix
        s = self.labels[l].intersection(self.labels[k])
        if len(s) > 0:
          sys.exit(f'{len(s)} non-unique labels for the label keys {l} and {k}, exiting now')
    if pickleDir is not None:
      self.pickleDir = pickleDir
      self.pickleOrRead(limitReads, plotSquiggle)
    relNucComp(self.readSummaryStats)
    nucleotideHistogram(self.readSummaryStats)
  # Extract summary stats from pickle or read from actual reads
  def pickleOrRead(self, limitReads = None, plotSquiggle = None):
    fname = self.pickleDir + '/summaryStats.pandas'
    if exists(fname):
      self.readSummaryStats = pandas.read_pickle(fname)
    else:
      pds = []
      cnt = 0
      for path in self.reads:
        print (f'READ PATH: {path}')
        for rname in Path(path).rglob(f'*.fast5'):
          cnt += 1
          if limitReads is not None and int(limitReads) < cnt:
            break
          print(f'{cnt: >4} {rname}')
          pd = Fast5.fast5Handler(self.labels,rname, plotSquiggle)
          pds.append(pd)
      self.readSummaryStats = pandas.concat(pds)
      self.readSummaryStats.to_pickle(fname)


# Jannes has extracted the IDs which correspond to 0%, 30%, 100% deuterium. However, for now we only
# care about 0% vs 100% deuterium

def getIdLabels(fname):
  labels=set()
  with open(fname, 'r') as r:
    for line in r:
      labels.add(line.strip())
  return labels

# Calculates the relative nucleotide composition of a read.
# TODO move the plotting functionality somewhere else!

def relNucComp(pd):
  ks1 = set()
  ks2 = set()
  ks3 = set()
  for n in pd['nstats']:
    for k in n.k1.keys():
      ks1.add(k)
    for k in n.k2.keys():
      ks2.add(k)
    for k in n.k3.keys():
      ks3.add(k)
  ks1=list(ks1)
  ks1.sort()
  ks2=list(ks2)
  ks2.sort()
  ks3=list(ks3)
  ks3.sort()
  print(ks1)
  print(ks2)
  print(ks3)
  xs1 = []
  xs2 = []
  xs3 = []
  for n in pd['nstats']:
    s1 = sum(n.k1.values())
    s2 = sum(n.k2.values())
    s3 = sum(n.k3.values())
    arr1 = np.array([n.k1[k] / s1 for k in ks1])
    arr2 = np.array([n.k2[k] / s2 for k in ks2])
    arr3 = np.array([n.k3[k] / s3 for k in ks3])
    xs1.append(arr1)
    xs2.append(arr2)
    xs3.append(arr3)
  pd['n1rel'] = xs1
  pd['n2rel'] = xs2
  pd['n3rel'] = xs3
  xs1bs = np.vstack(pd['n1rel'][pd['labels']==0])
  xs1dt = np.vstack(pd['n1rel'][pd['labels']==1])
  _, axes = plt.subplots(2,4, figsize=(40,10))
  for c,lbl in enumerate(ks1):
    ax = axes[0,c]
    ax.set_xlim(0.1,0.4)
    az.plot_posterior({lbl + ' (H2O)': xs1bs[:,c]},ax=ax, textsize=26)
    ax = axes[1,c]
    ax.set_xlim(0.1,0.4)
    az.plot_posterior({lbl + ' (D2O)': xs1dt[:,c]},ax=ax, textsize=26)
  plt.savefig('nucleotide-distributions.pdf', bbox_inches='tight')
  return


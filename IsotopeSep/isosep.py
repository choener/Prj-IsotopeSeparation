#!/usr/bin/env python3

from os.path import exists
from pathlib import Path
from pymc import Model, Normal, HalfCauchy, sample, Dirichlet, HalfNormal
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import aesara as ae
import aesara.tensor as at
import argparse
import arviz as az
import logging
import logging as log
import matplotlib as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pandas
import pickle
import pymc as mc

import Construct
import Fast5

font = { 'size': 10 }
#font = { 'weight': 'bold', 'size': 10 }
pl.rc('font', **font)

# 
#
# NOTE Reading 'reads' is costly only the first run, we pickles immediately, then re-use the pickles

def main ():
  FORMAT = '%(asctime)s %(message)s'
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  logging.info(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('--limitreads', help='Limit the number of reads to read when no pickle exists')
  parser.add_argument('--pickle', default="./tmp.pickle", help='where to write pickle data to')
  parser.add_argument('--plotsquiggle', help='Plot the time series squiggle plot for every (!) read')
  parser.add_argument('--reads', action='append', help='directories where reads are located')
  args = parser.parse_args()
  #
  # fill infrastructure for data
  construct = Construct.Construct(barcodes = args.barcode)
  # check if we have something to load, if so do that
  if exists(args.pickle):
    loaded = Construct.Construct.load(args.pickle)
    construct.merge(loaded)
    pass
  totReads = 0
  limitReads = int(args.limitreads)
  # Extract and process reads from files. Will save the current construct after every read. Should
  # be safe to Ctrl-C out of.
  for path in args.reads:
    log.info(f'READ PATH: {path}')
    for rname in Path(path).rglob(f'*.fast5'):
      if args.limitreads is not None and totReads >= limitReads:
        break
      log.info(f'FILE PATH" {rname}')
      cnt = construct.handleReadFile(rname, limitReads)
      totReads += cnt
      construct.save(args.pickle)
  log.info(f'Model loaded with {len(construct)} reads')






  #labels={}
  #labels['0'] = getIdLabels('barcode14.ids')
  #labels['30'] = getIdLabels('barcode15.ids')
  #labels['100'] = getIdLabels('barcode16.ids')
  #dir = '/shared/choener/Workdata/heavy_atoms_deuterium_taubert/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #dir = '.'
  #dir = '/data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing'
  #pdAll = None
  #if exists('./pdAll.pandas'):
  #  pdAll = pandas.read_pickle('./pdAll.pandas')
  #else:
  #  pds = []
  #  cnt = 0
  #  for path in Path(dir).rglob(f'*.fast5'):
  #    pd = fast5Handler(labels,path)
  #    pds.append(pd)
  #    cnt += 1
  #    #if cnt >= 10: break
  #  allpds = pandas.concat(pds)
  #  allpds.to_pickle('./pdAll.pandas')
  #  pdAll = allpds
  #relNucComp(pdAll)
  #nucleotideHistogram(pdAll)
  ## only work with data that has known labels
  #pdd = pdAll[pdAll['labels'] != -1]
  #pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  #pd, pdmean, pdstddev = normalize(pdd)
  ## TODO split off train / test
  #print(len(pd))
  #pd = pd.sample(frac=1)
  ##modelDirichletNucs(pd)
  #modelMixtureNucs(pd, k=1)
  #modelMixtureNucs(pd, k=2)
  #modelLogistic(pd, k=1)
  #modelLogistic(pd, k=2)
  ##testModel (pd[splitPoint:], mdl)

if __name__ == "__main__":
  main()


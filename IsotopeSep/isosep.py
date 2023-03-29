#!/usr/bin/env python3

from os.path import exists, isdir, isfile
import argparse
import logging
import logging as log
import matplotlib as pl
import pandas as pd
import pymc as mc
from pathlib import Path
import glob

import Construct
import Log

font = { 'size': 10 }
#font = { 'weight': 'bold', 'size': 10 }
pl.rc('font', **font)

# Simple main system
#
# NOTE Reading 'reads' is costly only the first run, we pickles immediately, then re-use the pickles

def main ():
  FORMAT = '%(asctime)s %(message)s'
  logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename='isosep.log', filemode='a')
  logging.info(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('--outputdir', default="tmp", help='where to write output and state data to')
  parser.add_argument('--summarydirs', action='append', help='directories where read pickles are located, or individual read pickles')
  parser.add_argument('--dataplots', default=False, action='store_true', help='actually run plots')
  parser.add_argument('--kmer', default='k1', help='k-mer length: k1, k3, k5 are legal')
  parser.add_argument('--train', default=False, action='store_true', help='enable Bayesian training')
  parser.add_argument('--posteriorpredictive', default=False, action='store_true', help='enable Bayesian posterior predictive')
  parser.add_argument('--priorpredictive', default=False, action='store_true', help='Prior predictive')
  args = parser.parse_args()
  # checks
  if args.barcode is None:
    log.error('no barcodes given')
    exit(0)
  for _, bc in args.barcode:
    if not exists(bc):
      log.error('{bc} does not exist')
  if not exists (args.outputdir):
    log.error(f'output directory "{args.outputdir}" does not exist')
    exit(0)
  if args.summarydirs is None:
    log.error('no summary.csv.zst given')
    exit(0)
  # prepare construct
  construct = Construct.Construct(barcodes = args.barcode)
  # loads all pickles
  # is directory
  # TODO
  csvs = []
  for p in args.summarydirs:
    log.info(f'PATH" {p}')
    if isfile(p):
      log.info(f'FILE PATH" {p}')
      df = pd.read_csv(p)
      csvs.append(df)
      #loaded = Construct.Construct.load(p)
      #construct.merge(loaded)
    if isdir(p):
      log.info(f'DIRECTORY PATH" {p}')
      for rname in Path(p).rglob(f'summary.csv.zst'):
        log.info(f'FILE PATH" {rname}')
        df = pd.read_csv(rname)
        csvs.append(df)
        #loaded = Construct.Construct.load(rname)
        #construct.merge(loaded)
  df = pd.concat(csvs)
  print(df)
  print(df.memory_usage())

  log.info(f'Model loaded with {len(construct)} reads')
  if (args.dataplots):
    assert(construct.summaryStats is not None)
    construct.summaryStats.postFile(args.outputdir)
  if args.train:
    assert(construct.summaryStats is not None)
  if args.posteriorpredictive:
    assert(construct.summaryStats is not None)
  if args.train or args.posteriorpredictive or args.priorpredictive:
    assert(construct.summaryStats is not None)
    Log.runModel(construct.summaryStats, kmer = args.kmer, train = args.train, posteriorpredictive = args.posteriorpredictive, priorpredictive = args.priorpredictive)



if __name__ == "__main__":
  #jax.default_backend()
  main()


#!/usr/bin/env python3

from os.path import exists, isdir, isfile
import argparse
import logging
import logging as log
import matplotlib as pl
import pandas as pandas
import pymc as mc
from pathlib import Path

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
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  logging.info(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('--outputdir', default="tmp", help='where to write output and pickle data to')
  parser.add_argument('--pickledreads', action='append', help='directories where read pickles are located, or individual read pickles')
  parser.add_argument('--plots', default=False, action='store_true', help='actually run plots')
  parser.add_argument('--kmer', default='k1', help='k-mer length: k1, k3, k5 are legal')
  parser.add_argument('--trainbayes', default=False, action='store_true', help='enable Bayesian training')
  parser.add_argument('--predictbayes', default=False, action='store_true', help='enable Bayesian posterior predictive')
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
  if args.pickledreads is None:
    log.error('no pickled reads given')
    exit(0)
  # prepare construct
  construct = Construct.Construct(barcodes = args.barcode)
  # loads all pickles
  # is directory
  # TODO
  for p in args.pickledreads:
    log.info(f'PATH" {p}')
    if isfile(p):
      log.info(f'FILE PATH" {p}')
      loaded = Construct.Construct.load(p)
      construct.merge(loaded)
    if isdir(p):
      log.info(f'DIRECTORY PATH" {p}')
      for rname in Path(p).rglob(f'*.pickle'):
        log.info(f'FILE PATH" {rname}')
        loaded = Construct.Construct.load(rname)
        construct.merge(loaded)

  log.info(f'Model loaded with {len(construct)} reads')
  if (args.plots):
    assert(construct.summaryStats is not None)
    construct.summaryStats.postFile(args.outputdir)
  if args.trainbayes:
    assert(construct.summaryStats is not None)
    Log.runModel(construct.summaryStats, kmer = args.kmer)
  if args.predictbayes:
    assert(construct.summaryStats is not None)
    # TODO Log.runModel, but with Bayesian posterior predictives



if __name__ == "__main__":
  main()


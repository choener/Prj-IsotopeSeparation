#!/usr/bin/env python3

from os.path import exists
from pathlib import Path
import argparse
import logging
import logging as log
import matplotlib as pl
import os
import pandas as pandas
import pymc as mc

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
  parser.add_argument('--limitreads', help='Limit the number of reads to read when no pickle exists')
  parser.add_argument('--outputs', default="tmp", help='where to write output and pickle data to')
  parser.add_argument('--reads', action='append', help='directories where reads are located, or single read')
  parser.add_argument('--plots', default=False, action='store_true', help='actually run plots')
  parser.add_argument('--kmer', default='k1', help='k-mer length: k1, k3, k5 are legal')
  parser.add_argument('--disableBayes', default=False, action='store_true', help='disable Bayesian modeling, only imports reads')
  parser.add_argument('--sideloadPickle', action='append', help='will import pickle files')
  parser.add_argument('--picklePath', default='construct.pickle')
  args = parser.parse_args()
  #
  # fill infrastructure for data
  if not exists (args.outputs):
    os.mkdir(args.outputs)
  construct = Construct.Construct(barcodes = args.barcode)
  # check if we have something to load, if so do that
  picklePath = os.path.join(args.outputs, args.picklePath)
  if exists(picklePath):
    loaded = Construct.Construct.load(picklePath)
    construct.merge(loaded)
  # if there is a multitude of pickles, then load those
  if args.sideloadPickle is not None:
    for p in Path(args.sideloadPickle).rglob(f'*.pickle'):
      construct.merge(p)
  totReads = 0
  limitReads = int(args.limitreads)
  # Extract and process reads from files. Will save the current construct after every read. Should
  # be safe to Ctrl-C out of.
  for path in args.reads:
    log.info(f'READ PATH: {path}')
    if os.path.isfile(path):
      log.info(f'FILE PATH" {path}')
      cnt = construct.handleReadFile(path, limitReads)
      totReads += cnt
      construct.save(picklePath)
    for rname in Path(path).rglob(f'*.fast5'):
      if args.limitreads is not None and totReads >= limitReads:
        break
      log.info(f'FILE PATH" {rname}')
      cnt = construct.handleReadFile(rname, limitReads)
      totReads += cnt
      construct.save(picklePath)
  log.info(f'Model loaded with {len(construct)} reads')
  if (args.plots):
    assert(construct.summaryStats is not None)
    construct.summaryStats.postFile(args.outputs)
  # TODO create data frame
  # TODO run stats model
  if not args.disableBayes:
    assert(construct.summaryStats is not None)
    Log.runModel(construct.summaryStats, kmer = args.kmer)



if __name__ == "__main__":
  main()


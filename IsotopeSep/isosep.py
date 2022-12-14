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
  parser.add_argument('--reads', action='append', help='directories where reads are located')
  parser.add_argument('--plots', default=False, action='store_true', help='actually run plots')
  args = parser.parse_args()
  #
  # fill infrastructure for data
  if not exists (args.outputs):
    os.mkdir(args.outputs)
  construct = Construct.Construct(barcodes = args.barcode)
  picklePath = os.path.join(args.outputs, "construct.pickle")
  # check if we have something to load, if so do that
  if exists(picklePath):
    loaded = Construct.Construct.load(picklePath)
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
      construct.save(picklePath)
  log.info(f'Model loaded with {len(construct)} reads')
  assert(construct.summaryStats is not None)
  if (args.plots):
    construct.summaryStats.postFile(args.outputs)
  # TODO create data frame
  # TODO run stats model
  Log.runModel()



if __name__ == "__main__":
  main()


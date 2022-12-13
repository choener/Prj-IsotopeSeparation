#!/usr/bin/env python3

from os.path import exists
from pathlib import Path
import argparse
import logging
import logging as log
import matplotlib as pl
import pandas as pandas
import pymc as mc

import Construct

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
  parser.add_argument('--pickle', default="./tmp.pickle", help='where to write pickle data to')
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



if __name__ == "__main__":
  main()


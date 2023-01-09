#!/usr/bin/env python3

# Prepares data into many pickle files. isosep.py then reads those pickles to do the actual
# stats-modeling work.


from os.path import exists, basename, splitext
import argparse
import logging
import logging as log
import os
import pandas as pandas
import pymc as mc

import Construct


""" This small tool will go through a single read and extract, then pickle, the main information.
"""

def main ():
  FORMAT = '%(asctime)s %(message)s'
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  logging.info(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('-l', '--limitreads', help='Limit the number of reads to read when no pickle exists')
  parser.add_argument('-o', '--outputdir', default="tmp", help='where to write output pickle data to, the pickle file name is derivide from the input name')
  parser.add_argument('-i', '--input', help='single read to process')
  args = parser.parse_args()
  #
  # fill infrastructure for data
  if args.barcode is None:
    log.error('no barcodes given')
    exit(0)
  for _, bc in args.barcode:
    if not exists(bc):
      log.error('{bc} does not exist')
  if not exists (args.outputdir):
    log.error(f'output directory "{args.outputdir}" does not exist')
    exit(0)
  if args.input is None:
    log.error('no input was given')
    exit(0)
  construct = Construct.Construct(barcodes = args.barcode)
  input = args.input
  inputbase, _ = splitext(args.input)
  output = os.path.join(args.outputdir, basename(inputbase)) + '.pickle'
  log.info(f'read file is {input}, output will be written to {output}')
  # check if we have something to load, if so do that
  if exists(output):
    log.info(f'{output} exists, loading ...')
    loaded = Construct.Construct.load(output)
    construct.merge(loaded)
  limitReads = int(args.limitreads)
  cnt = construct.handleReadFile(input, limitReads)
  log.info(f'{output}: read {cnt} reads')
  construct.save(output)
  log.info(f'{output} has been saved')



if __name__ == "__main__":
  main()


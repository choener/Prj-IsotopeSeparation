#!/usr/bin/env python3

from os.path import exists, isdir, isfile, join, dirname
from pathlib import Path
import argparse
import logging
import logging as log
import matplotlib as pl
import pandas as pd
import pymc as mc
from hashlib import sha512

import Construct
import Log

font = { 'size': 10 }
#font = { 'weight': 'bold', 'size': 10 }
pl.rc('font', **font)

"""
Simple main system. Sets up the command-line parser, reads input barcode data, and summary data.

NOTE Reading 'reads' is costly only the first run, we pickles immediately, then re-use the pickles
"""

def main ():
  FORMAT = '%(asctime)s %(message)s'
  logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename='isosep.log', filemode='a')
  logging.info(f'PyMC v{mc.__version__}')
  parser = argparse.ArgumentParser()
  parser.add_argument('--logstderr', action='store_true', default=False, help='log to stderr as well')
  parser.add_argument('--barcode', action='append', nargs='+', help='given as PERCENT FILE')
  parser.add_argument('--outputdir', default="tmp", help='where to write output and state data to')
  parser.add_argument('--summarydirs', action='append', help='directories where read pickles are located, or individual read pickles')
  parser.add_argument('--dataplots', default=False, action='store_true', help='actually run plots')
  parser.add_argument('--kmer', default='1', help='k-mer length: 1, 3, 5 are assumed available')
  parser.add_argument('--train', default=False, action='store_true', help='enable Bayesian training')
  parser.add_argument('--posteriorpredictive', default=False, action='store_true', help='enable Bayesian posterior predictive')
  parser.add_argument('--priorpredictive', default=False, action='store_true', help='Prior predictive')
  parser.add_argument('--maxsamples', default=None, help='restrict number of samples to train on')
  parser.add_argument('--sampler', default="advi-nuts", choices=['adagrad','advi','jax','nuts','advi-nuts'], help='choice of sampler')
  args = parser.parse_args()
  if args.logstderr is True:
    logging.getLogger().addHandler(logging.StreamHandler())
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
  # prepare construct: we store an efficient pickle of the data we work with.
  # TODO summarydirs should be a list of directories that can be handed over ...
  hashstore = sha512((args.kmer + str(args.summarydirs)).encode('utf-8')).hexdigest()
  if not exists ("./store"):
    log.error(f'store directory does not exist')
    exit(0)
  storename = join("./store", hashstore + ".pickle")
  construct = Construct.Construct()
  for p,b in args.barcode:
    construct.addbarcode(p,b)
  if exists(storename):
    construct.loadgroups(storename)
  else:
    for p in args.summarydirs:
      log.info(f'PATH" {p}')
      if isfile(p):
        log.info(f'FILE PATH" {p}')
        df = pd.read_csv(p)
        rds = pd.read_csv(join(dirname(p),"reads.csv.zst"))
        construct.addkmerdf(args.kmer, df, rds)
      if isdir(p):
        log.info(f'DIRECTORY PATH" {p}')
        for rname in Path(p).rglob(f'summary.csv.zst'):
          log.info(f'FILE PATH" {rname}')
          df = pd.read_csv(rname)
          rds = pd.read_csv(join(dirname(rname),"reads.csv.zst"))
          construct.addkmerdf(args.kmer, df, rds)
    construct.mergegroups()
    construct.savegroups(storename)

  log.info(f'Model loaded with {len(construct)} reads')
  if args.train or args.posteriorpredictive or args.priorpredictive:
    Log.runModel(args.kmer, construct.dfgroups[0], train = args.train, posteriorpredictive = args.posteriorpredictive, priorpredictive = args.priorpredictive, maxsamples = args.maxsamples, sampler = args.sampler)



if __name__ == "__main__":
  #jax.default_backend()
  main()


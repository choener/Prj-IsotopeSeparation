#!/usr/bin/env python3

from os.path import exists, isfile, join, dirname, split
import argparse
import logging
import logging as log
import matplotlib as pl
import pandas as pd
import pymc as mc
from hashlib import sha512
import glob
import gc

import Construct
import Log

font = {'size': 10}
# font = { 'weight': 'bold', 'size': 10 }
pl.rc('font', **font)


"""
Simple main system. Sets up the command-line parser, reads input barcode data, and summary data.

NOTE Reading 'reads' is costly only the first run, we pickles immediately, then re-use the pickles
"""


def main():
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        filename='isosep.log', filemode='a')
    logging.info(f'PyMC v{mc.__version__}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--logstderr', action='store_true',
                        default=False, help='log to stderr as well')
    parser.add_argument('--barcode', action='append',
                        nargs='+', help='given as PERCENT FILE')
    parser.add_argument('--outputdir', default="tmp",
                        help='where to write output and state data to')
    parser.add_argument('--inputdirs', nargs='+',
                        action='append', help='directories with input files')
    parser.add_argument('--dataplots', default=False,
                        action='store_true', help='actually run plots')
    parser.add_argument('--kmer', default='5',
                        help='k-mer length: 1, 3, 5 are assumed available')
    parser.add_argument('--train', default=False,
                        action='store_true', help='enable Bayesian training')
    parser.add_argument('--posteriorpredictive', default=False,
                        action='store_true', help='enable Bayesian posterior predictive')
    parser.add_argument('--priorpredictive', default=False,
                        action='store_true', help='Prior predictive')
    parser.add_argument('--maxsamples', default=None,
                        help='restrict number of samples to train on')
    parser.add_argument('--sampler', default="advi-nuts", choices=[
                        'adagrad', 'advi', 'jax', 'nuts', 'advi-nuts'], help='choice of sampler')
    parser.add_argument('--zero', default=0.0,
                        help='relative abundance mapped to False')
    parser.add_argument('--one', default=1.0,
                        help='relative abundance mapped to True')
    parser.add_argument('--onlycomplete', default=False, action='store_true',
                        help='will remove incomplete reads, say for training')
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
    if not exists(args.outputdir):
        log.error(f'output directory "{args.outputdir}" does not exist')
        exit(0)
    if args.inputdirs is None:
        log.error('no summary.csv.zst given')
        exit(0)
    # collect all paths that contain the necessary files.
    inputs = []
    for ps in args.inputdirs:
        for p in ps:
            findstr = f'**.{args.kmer}.pickle.zst'
            print(p, findstr)
            for d in glob.glob(join(p, findstr), recursive=True):
                inputs.append(d)
    # need to load barcodes now
    barcodes = pd.DataFrame()
    for p, b in args.barcode:
        print(p, b)
        df = pd.read_csv(b, header=None)
        df.columns = ['read']
        df['p'] = float(p) / 100
        df = df.set_index('read')
        barcodes = barcodes.append(df)
    constructs = []
    for i in inputs:
        construct = Construct.Construct(args.kmer)
        construct.load(i)
        construct.addfilterbarcodes(barcodes)
        if args.onlycomplete:
            construct.onlycomplete()
        # TODO Need to remove from construct.df those reads that are not part of the any barcode
        constructs.append(construct)

    log.info(
        f'Model data loaded')
    # TODO make sure to select correct targets
    if args.train or args.posteriorpredictive or args.priorpredictive:
        Log.runModel(args.zero, args.one, args.outputdir, args.kmer,
                     constructs, train=args.train, posteriorpredictive=args.posteriorpredictive, priorpredictive=args.priorpredictive, maxsamples=args.maxsamples, sampler=args.sampler)


if __name__ == "__main__":
    # jax.default_backend()
    main()

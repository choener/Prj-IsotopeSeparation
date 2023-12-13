#!/usr/bin/env python3

# Prepares data into many pickle files. isosep.py then reads those pickles to do the actual
# stats-modeling work.


from os.path import exists, isfile, join, dirname, split
from os.path import exists, basename, splitext
import argparse
import logging
import logging as log
import os
import pandas as pandas
import pymc as mc
import pandas as pd

import Construct


"""
This small tool will go through a single read file and extract, then pickle, the main information. Extract files are based on a special k-mer number each and contain all the necessary statistics, apart from barcode information.
"""

def main ():
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logging.info(f'PyMC v{mc.__version__}')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputdir', default="tmp", help='where to write output pickle data to, the pickle file name is derived from the input name')
    parser.add_argument('-i', '--input', help='single read file to process')
    parser.add_argument('--kmer', default='5', help='k-mer length: 1, 3, 5 are assumed available')
    args = parser.parse_args()
    if not exists (args.outputdir):
        log.error(f'output directory "{args.outputdir}" does not exist')
        exit(0)
    if args.input is None:
        log.error('no input was given')
        exit(0)
    input = args.input
    inputbase, _ = splitext(args.input)
    output = os.path.join(args.outputdir, basename(inputbase)) + '.' + args.kmer + '.pickle.zst'
    summaryFile = join(input,'summary.csv.zst')
    readsFile = join(input,'reads.csv.zst')
    log.info(f'read file is {input}, output will be written to {output}')
    # TODO This is how isosep.py reads in files, do this here too
    sumdf = pd.read_csv(summaryFile)
    rdsdf = pd.read_csv(readsFile)
    construct = Construct.Construct(args.kmer)
    construct.addkmerdf(args.kmer, sumdf, rdsdf)
    cnt = len(construct.df)
    log.info(f'{output}: read {cnt} reads')
    construct.save(output)
    log.info(f'{output} has been saved')



if __name__ == "__main__":
  main()


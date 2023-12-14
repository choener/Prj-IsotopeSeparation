#!/usr/bin/env python3

# Small tool that generates a read statistics CSV for each fast5 read. These CSV are then consumed
# by the main isosep.py program.

import os
import argparse
import pandas as pd
import numpy as np
import sys
from csv import writer
import signal
import zstandard

import Fast5
import Stats

# exit_gracefully = False
#
# def signal_handler(sig, frame):
#  print('Ctrl+C captured. Prepare to exit gracefully...')
#  exit_gracefully = True
#
# signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--outdir', help='csv with the output data. defaults to input with csv suffix')
parser.add_argument('input', nargs=1)
parser.add_argument(
    '--maxcount', help='maximal number of reads to process at once')

args = parser.parse_args()

infname = args.input[0]
outdir = infname
outdir = os.path.basename(outdir)
outdir = os.path.splitext(outdir)[0]
if args.outdir is not None:
    outdir = args.outdir

if infname == outdir:
    print(f'{infname} and {outdir} are the same file, aborting')
    exit(1)

os.makedirs(outdir, exist_ok=True)


# summary information for reads. This is one line per read only
readsfile = os.path.join(outdir, "reads.csv.zst")
reads = pd.DataFrame()
if os.path.exists(readsfile):
    reads = pd.read_csv(readsfile)
    print('reading readsfile')
assert reads is not None

summaryfile = os.path.join(outdir, "summary.csv.zst")

"""
Returns the statistics for a k-mer based on the whole read information.
"""

def genKx(r, readpd, kx):
    rtrn = []
    for k,g in readpd.groupby(kx):
        medianMed, medianMad = Stats.medianMad(g['medianZ'])
        dwellMean = np.mean(g['dwellTime'])
        rtrn.append({'read': r, 'k': k, 'medianZ': medianMed, 'madZ': medianMad, 'dwellMean': dwellMean})
    #rtrn = []
    #for k in sorted(set(readpd[kx])):
    #    medianMed, medianMad = Stats.medianMad(
    #        readpd[readpd[kx] == k]['medianZ'])
    #    #dwellMed, dwellMad = Stats.medianMad(
    #    #    readpd[readpd[kx] == k]['dwellTime'])
    #    dwellMean = np.mean(readpd[readpd[kx] == k]['dwellTime'])
    #    rtrn.append({'read': r, 'k': k, 'medianZ': medianMed, 'madZ': medianMad, 'dwellMean': dwellMean})
    return rtrn

# Collect statistics per read and for each read also the kmer statistics


readIDs = Fast5.fast5Reads(infname)
numR = len(readIDs)
for cnt, r in enumerate(readIDs):
    r = sys.intern(r)

    # TODO handle sigint
    if args.maxcount is not None and cnt >= int(args.maxcount):
        break

    # ignore known reads
    # BUG ignore reads based on ID in the future!
    if len(reads) > 0 and (cnt < len(reads) or any(reads['read'] == r)):
        print(f'IGNORE [{cnt :5d} / {numR : 5d}] {r}')
    else:
        print(f'NEW [{cnt :5d} / {numR : 5d}] {r}')
        preRaw, sufRaw, segmented, nucs = Fast5.fast5ReadData(infname, r)
        medianR, madR = Stats.medianMad(sufRaw)
        pfxMedianR, pfxMadR = Stats.medianMad(preRaw)
        # this is O(n^2), but @n=4000@ over a long while
        new = pd.DataFrame()
        new = new.append({'read': r, 'median': medianR, 'mad': madR,
                             'pfxMedian': pfxMedianR, 'pfxMad': pfxMadR}, ignore_index=True)
        #reads.to_csv(readsfile, index=False, float_format='%.3f')
        with zstandard.open(readsfile, 'a') as rf:
            new.to_csv(rf, index=False, header=(
                cnt == 0), float_format='%.3f')

        # for each nucleotide position in each read ...
        # NOTE the range is only for valid kmers!
        readpd = pd.DataFrame()
        eachPos = []
        for i in range(2, len(nucs)-2):
            median, mad = Stats.medianMad(segmented[i])
            medianZ, madZ = Stats.medianMad((segmented[i]-medianR)/madR)
            eachPos.append(
                {'read': cnt, 'k1': nucs[i:i+1], 'k3': nucs[i-1:i+2], 'k5': nucs[i-2:i+3], 'median': median, 'mad': mad, 'medianZ': medianZ, 'madZ': madZ, 'dwellTime': len(segmented[i])
                 })
        readpd = readpd.append(eachPos, ignore_index=True)
        readpd = readpd.sort_values(by='k5') # will be sorted by all kX automatically then!
        k1 = genKx(r, readpd, 'k1')
        k3 = genKx(r, readpd, 'k3')
        k5 = genKx(r, readpd, 'k5')
        pdeach = pd.DataFrame(k1 + k3 + k5)
        with zstandard.open(summaryfile, 'a') as sf:
            pdeach.to_csv(sf, index=False, header=(
                cnt == 0), float_format='%.3f')
        # TODO Not writing out these anymore, since they are not used.
        #readpd = readpd.drop(['k1', 'k3'], axis=1)
        #readpd.to_csv(os.path.join(
        #    outdir, f'{r}.kmers.csv.zst'), index=False, float_format='%.3f')

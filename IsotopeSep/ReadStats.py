#!/usr/bin/env python3

# Small tool that generates a read statistics CSV for each fast5 read. These CSV are then consumed
# by the main isosep.py program.

import os
import argparse
import pandas as pd
import numpy as np
import sys

import Fast5
import Stats



parser = argparse.ArgumentParser()
parser.add_argument('--outdir', help='csv with the output data. defaults to input with csv suffix')
parser.add_argument('input', nargs=1)

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

readsfile = ""

summs = []
reads = pd.DataFrame()
if os.path.exists(readsfile):
  pd.read_csv(readsfile)

eachRead = []

def appendKx(eackK, r, readpd, kx):
  for k in sorted(set(readpd[kx])):
    medianMed, medianMad = Stats.medianMad(readpd[readpd[kx]==k]['medianZ'])
    eachK.append({'read': r, 'k': k, 'medianZ': medianMed, 'madZ': medianMad})

# Collect statistics per read and for each read also the kmer statistics

readIDs = Fast5.fast5Reads(infname)
numR = len(readIDs)
for cnt, r in enumerate(readIDs):
  r = sys.intern(r)
  print(f'[{cnt :5d} / {numR : 5d}] {r}')
  preRaw, sufRaw, segmented, nucs = Fast5.fast5ReadData(infname, r)
  medianR, madR = Stats.medianMad(sufRaw)
  eachRead.append({'read': r, 'median': medianR, 'mad': madR})
  #for each nucleotide position in each read ...
  #NOTE the range is only for valid kmers!
  readpd = pd.DataFrame()
  eachPos = []
  for i in range(2,len(nucs)-2):
    median, mad = Stats.medianMad(segmented[i])
    medianZ, madZ = Stats.medianMad((segmented[i]-medianR)/madR)
    eachPos.append(
        { 'read': cnt
        , 'k1': nucs[i:i+1], 'k3': nucs[i-1:i+2], 'k5': nucs[i-2:i+3]
        , 'median': median, 'mad': mad
        , 'medianZ': medianZ, 'madZ': madZ
         })
  readpd = readpd.append(eachPos, ignore_index=True)
  readpd.to_csv(os.path.join(outdir, f'{r}.kmers.csv.zst'), index=False)
  eachK = []
  appendKx(eachK, r, readpd, 'k1')
  appendKx(eachK, r, readpd, 'k3')
  appendKx(eachK, r, readpd, 'k5')
  pdeach = pd.DataFrame()
  pdeach = pdeach.append(eachK)
  summs.append(pdeach)
  # TODO append in linear time to readspd ...

summspd = pd.DataFrame().append(summs, ignore_index=True)
summspd.to_csv(os.path.join(outdir, "summary.csv.zst"), index=False)

readspd = reads.append(eachRead, ignore_index=True)
readspd.to_csv(os.path.join(outdir, "reads.csv.zst"), index=False)


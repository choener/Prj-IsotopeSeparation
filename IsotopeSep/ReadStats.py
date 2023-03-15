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
parser.add_argument('--outfile', help='csv with the output data. defaults to input with csv suffix')
parser.add_argument('input', nargs=1)

args = parser.parse_args()

infname = args.input[0]
outfname = infname
outfname = os.path.basename(outfname)
outfname = os.path.splitext(outfname)[0]
if args.outfile is not None:
  outfname = args.outfile

if infname == outfname:
  print(f'{infname} and {outfname} are the same file, aborting')
  exit(1)

# append to existing csv?
if os.path.exists(outfname):
  print(f'{outfname} already exists')
  exit(0)

print(infname)
print(outfname)

kmers = []
summs = []
reads = pd.DataFrame()

eachRead = []

def appendKx(eackK, r, readpd, kx):
  for k in sorted(set(readpd[kx])):
    medianMed, medianMad = Stats.medianMad(readpd[readpd[kx]==k]['medianZ'])
    eachK.append({'read': r, 'k': k, 'medianZ': medianMed, 'madZ': medianMad})

# Collect statistics per read and for each read also the kmer statistics

readIDs = Fast5.fast5Reads(infname)
numR = len(readIDs)
cnt = 0
for r in readIDs:
  cnt = cnt+1
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
        { 'read': r
        , 'k1': nucs[i:i+1], 'k3': nucs[i-1:i+2], 'k5': nucs[i-2:i+3]
        , 'median': median, 'mad': mad
        , 'medianZ': medianZ, 'madZ': madZ
         })
  readpd = readpd.append(eachPos, ignore_index=True)
  kmers.append(readpd)
  eachK = []
  appendKx(eachK, r, readpd, 'k1')
  appendKx(eachK, r, readpd, 'k3')
  appendKx(eachK, r, readpd, 'k5')
  pdeach = pd.DataFrame()
  pdeach = pdeach.append(eachK)
  summs.append(pdeach)

reads = reads.append(eachRead, ignore_index=True)
kmers = pd.DataFrame().append(kmers, ignore_index=True)
summs = pd.DataFrame().append(summs, ignore_index=True)
summs.to_csv(outfname + ".summary.csv.zst")
kmers.to_csv(outfname + ".kmers.csv.zst")
reads.to_csv(outfname + ".reads.csv.zst")


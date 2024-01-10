#!/usr/bin/env python3

# Reads in a number of *csv files to generate cross-validation plots.
# Each directory given is assumed to be one instance of cross-validation.

from os.path import exists, isfile, join, dirname, split
import argparse
import logging
import logging as log
import matplotlib as pl
import matplotlib.pyplot as plt
import pandas as pd
import pymc as mc
import glob
from time import gmtime, strftime
import arviz as az
import numpy as np

az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 1000
fontsz = 12  # fontsize
titlesz = 16  # fontsize for title
linew = 2  # line width of important lines
cOrange = '#d95f02'
cBlue = '#7570b3'
cGreen = '#1b9e77'

# Plots the FDR for all members of the cross-validation set.

def plotFDR(fs, k):
    #df = pd.DataFrame({'cutoff': rs, 'fdr': ys, 'relreads': ns})
    #df.to_csv(f'{fnamepfx}-fdr.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.grid(c='grey')
    ax2 = ax1.twinx()
    ax1.set_facecolor('white')
    ax1.set_xlabel('Cutoff', fontsize=fontsz)
    ax2.set_ylabel('FDR', fontsize=fontsz)
    ax2.set_title('False discovery rate', fontsize=titlesz)
    ax2.grid(None)
    ax1.set_ylabel('Fraction of reads', fontsize=fontsz)
    for i,f in enumerate(fs):
        df = pd.read_csv(join(f,f'{k}-adagrad-fdr.csv'))
        lbl = None
        if i == 0:
            lbl = f'FDR'
        ax2.plot(df.cutoff, df.fdr, color='black', label=lbl)
        lbl = None
        if i == 0:
            lbl = f'Fraction of reads'
        ax1.plot(df.cutoff, df.relreads, color='blue', label=lbl)
    fig.legend(frameon=True, facecolor='white', framealpha=1.0,
               loc='lower right', bbox_to_anchor=(0.85, 0.15))
    plt.savefig(f'{k}-cross-fdr.png')
    plt.savefig(f'{k}-cross-fdr.pdf')
    plt.close()

def plotErrorResponse(fs, k, zeroLabel, oneLabel):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('white')
    plt.grid(c='grey')
    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=linew)
    p0s = []
    p1s = []
    for i, f in enumerate(fs):
        df = pd.read_csv(join(f,f'{k}-adagrad-response.csv'))
        p0good = len(list(filter(lambda x: (x<0.5), df.p0))) / len(df.p0)
        p1good = len(list(filter(lambda x: (x<0.5), df.p1))) / len(df.p1)
        p0s.append(p0good)
        p1s.append(p1good)
        xp0 = np.arange(len(df.p0)) / len(df.p0)
        lbl = None
        if i == 0:
            lbl = zeroLabel
        ax.plot(xp0, df.p0, color=cOrange,
                label=lbl, linewidth=linew)
        xp1 = np.arange(len(df.p1)) / len(df.p1)
        if i == 0:
            lbl = oneLabel
        ax.plot(xp1, df.p1, color=cBlue, label=lbl, linewidth=linew)
    # horizontal line at error 0.5
    ax.set_xlabel('Samples (ordered by distance)', fontsize=fontsz)
    ax.set_ylabel('Distance to true class (lower is better)', fontsize=fontsz)
    ax.set_title('Error response', fontsize=titlesz)
    # ax.legend(frameon=True, framealpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=1.0,
              loc='upper left', bbox_to_anchor=(0.1, 0.9))
    # averages !
    p0good = np.mean(p0s)
    p1good = np.mean(p1s)
    plt.axvline(x=p0good, color=cOrange, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p0good:.2f}',
                 xy=(p0good-0.1, 0.75), color=cOrange, fontsize=fontsz)
    plt.axvline(x=p1good, color=cBlue, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p1good:.2f}',
                 xy=(p1good+0.03, 0.25), color=cBlue, fontsize=fontsz)
    plt.savefig(f'{k}-model-error.png')
    plt.savefig(f'{k}-model-error.pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdirs', nargs='+',
                        action='append', help='directories with input files')
    parser.add_argument('--kmer')
    parser.add_argument('--zerolabel')
    parser.add_argument('--onelabel')
    args = parser.parse_args()
    ids = [f for fs in args.inputdirs for f in fs]
    print(ids)
    plotFDR(ids, args.kmer)
    plotErrorResponse(ids, args.kmer, args.zerolabel, args.onelabel)

if __name__ == "__main__":
    main()

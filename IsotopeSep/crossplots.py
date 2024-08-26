#!/usr/bin/env python3

# Reads in a number of *csv files to generate cross-validation plots.
# Each directory given is assumed to be one instance of cross-validation.

from collections import OrderedDict
from os.path import exists, isfile, join, dirname, split
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import numpy as np
from functools import reduce
import seaborn as sb
import arviz.labels as azl

az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 1000
fontsz = 18  # fontsize
titlesz = 26  # fontsize for title
linew = 3  # line width of important lines
cOrange = '#d95f02'
cBlue = '#7570b3'
cGreen = '#1b9e77'

# Plots the FDR for all members of the cross-validation set.

def plotFDR(fs, k, withmean, withstddev, withlines, fdrselection):
    #df = pd.DataFrame({'cutoff': rs, 'fdr': ys, 'relreads': ns})
    #df.to_csv(f'{fnamepfx}-fdr.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    plt.grid(color='lightblue', linestyle='--')
    ax2 = ax1.twinx()
    #ax1.grid(None)
    ax1.set_facecolor('white')
    ax1.set_xlabel('Cutoff', fontsize=fontsz)
    plt.grid(color='grey')
    ax2.set_ylabel('FDR', fontsize=fontsz)
    #ax2.set_title('False discovery rate', fontsize=titlesz)
    ax1.set_ylabel('% reads', fontsize=fontsz)
    # buckets for fdr between 0 and 0.5
    fdr = []
    relreads = []
    # individual lines, and data collection
    for f in fs:
        df = pd.read_csv(join(f,f'{k}-adagrad-fdr.csv'))
        fdr.append(df.fdr)
        relreads.append(df.relreads)
        if withlines:
            ax1.plot(df.cutoff, df.relreads, '.', color='blue', label=f'% reads')
            ax2.plot(df.cutoff, df.fdr, '.', color='black', label=f'FDR')
    # plot mean and standard deviation
    if withstddev:
        m = np.mean(fdr, axis=0)
        s = np.std(fdr, axis=0)
        ax2.fill_between(df.cutoff, m+s,np.fmax(0,m-s), color='black', alpha=0.3, label='FDR')
        m = np.mean(relreads, axis=0)
        s = np.std(relreads, axis=0)
        ax1.fill_between(df.cutoff, m+s,np.fmax(0,m-s), color='blue', alpha=0.3, label='% reads')
    if withmean:
        fdrmean = np.mean(fdr,axis=0)
        relreadsmean = np.mean(relreads,axis=0)
        ax2.plot(df.cutoff, fdrmean, color='black', label='FDR')
        ax1.plot(df.cutoff, relreadsmean, color='blue', label='% reads')
        # find index of largest fdr <= fdrselection and print cutoff and percent reads
        if fdrselection is not None:
            print(fdrmean)
            print(float(fdrselection))
            fdrindex = np.argmax (fdrmean >= float(fdrselection)) -1
            print(fdrmean <= float(fdrselection))
            print(fdrindex)
            print(f'fdr index: {fdrindex}, fdr: {fdrmean[fdrindex]}, Cutoff: {df.cutoff[fdrindex]}, Pcnt reads: {relreadsmean[fdrindex]}')
    # finish up
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels1 + labels2, handles1 + handles2))
    fig.legend(by_label.values(), by_label.keys(), frameon=True, facecolor='white', framealpha=1.0,
               loc='lower right', bbox_to_anchor=(0.85, 0.10), fontsize=fontsz)
    plt.xticks(fontsize = fontsz)
    plt.yticks(fontsize = fontsz)
    plt.savefig(f'{k}-cross-fdr.png')
    plt.savefig(f'{k}-cross-fdr.pdf')
    plt.close()

def plotErrorResponse(fs, k, zeroLabel, oneLabel, withmean, withstddev, withlines):
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('white')
    plt.grid(c='grey')
    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=linew)
    p0s = []
    p1s = []
    hist0s = []
    hist1s = []
    for i, f in enumerate(fs):
        df = pd.read_csv(join(f,f'{k}-adagrad-response.csv'))
        p0good = len(list(filter(lambda x: (x<0.5), df.p0))) / len(df.p0)
        p1good = len(list(filter(lambda x: (x<0.5), df.p1))) / len(df.p1)
        p0s.append(p0good)
        p1s.append(p1good)
        xp0 = np.arange(len(df.p0)) / len(df.p0)
        histDiv, _ = np.histogram(xp0, bins=200) # count number of elements per bin
        hist0, _ = np.histogram(xp0, bins=200, weights=df.p0) # sum up weights
        hist0 = hist0 / histDiv
        hist0s.append(hist0)
        if withlines:
            ax.plot(xp0, df.p0, color=cOrange,
                    label=zeroLabel, linewidth=linew)
        xp1 = np.arange(len(df.p1)) / len(df.p1)
        histDiv, _ = np.histogram(xp1, bins=200) # count number of elements per bin
        hist1, _ = np.histogram(xp1, bins=200, weights=df.p1) # sum up weights
        hist1 = hist1 / histDiv
        hist1s.append(hist1)
        if withlines:
            ax.plot(xp1, df.p1, color=cBlue, label=oneLabel, linewidth=linew)
    # mean and stddev
    if withstddev:
        m = np.mean(hist0s, axis=0)
        s = np.std(hist0s, axis=0)
        ax.fill_between(np.arange(0,1,0.005), m+s,np.fmax(0,m-s), color=cOrange, alpha=0.3, label=zeroLabel)
        m = np.mean(hist1s, axis=0)
        s = np.std(hist1s, axis=0)
        ax.fill_between(np.arange(0,1,0.005), m+s,np.fmax(0,m-s), color=cBlue, alpha=0.3, label=oneLabel)
    if withmean:
        ax.plot(np.arange(0,1,0.005), np.mean(hist0s, axis=0), color=cOrange, label=zeroLabel)
        ax.plot(np.arange(0,1,0.005), np.mean(hist1s, axis=0), color=cBlue, label=oneLabel)
    # horizontal line at error 0.5
    ax.set_xlabel('Relative number of reads', fontsize=fontsz)
    ax.set_ylabel('Distance to true class (lower is better)', fontsize=fontsz)
    #ax.set_title('Error response', fontsize=titlesz)
    # ax.legend(frameon=True, framealpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, facecolor='white', framealpha=1.0,
              loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=fontsz)
    # averages !
    p0good = np.mean(p0s)
    p1good = np.mean(p1s)
    plt.axvline(x=p0good, color=cOrange, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p0good:.2f}',
                 xy=(p0good-0.15, 0.75), color=cOrange, fontsize=fontsz)
    plt.axvline(x=p1good, color=cBlue, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p1good:.2f}',
                 xy=(p1good+0.03, 0.25), color=cBlue, fontsize=fontsz)
    plt.xticks(fontsize = fontsz)
    plt.yticks(fontsize = fontsz)
    plt.savefig(f'{k}-model-error.png')
    plt.savefig(f'{k}-model-error.pdf')
    plt.close()

# Generate common plots of best and worst kmers from the netcdf data
#        scaleMeans = abs(trace.posterior["scale"].mean(("chain", "draw")))
#        scaleZ = scaleMeans / trace.posterior["scale"].std(("chain", "draw"))
#        sortedScaleTrace = trace.posterior["scale"].sortby(scaleZ)
#        scaleCoords = scaleZ.sortby(scaleZ).coords['kmer'].values
#        # best 10 scale values
#        plotForest(fnamepfx, 'zsortedforest-scale-worst', kmer,
#                   sortedScaleTrace.sel(kmer=scaleCoords[:12]))

def plotForests(fs):
    ts = [ az.from_netcdf(join(f,f'5-adagrad-trace.netcdf')) for f in fs ]
    def go(high,low,tyname,ts):
        # first, construct kmers
        kmers = []
        for w in ["scale", "mad"]:
            scs = []
            for t in ts:
                means = abs(t.posterior[w].mean(("chain","draw")))
                z = means / 1 # t.posterior["scale"].std(("chain","draw"))
                s = t.posterior[w].sortby(z)
                c = z.sortby(z).coords['kmer'].values
                scs.append((s,c))
            # best
            # collect the *set* of kmers to use
            kmers = list(reduce(set.union, [set(c[high:]) for (_,c) in scs]))
            plotForest(f'{w}-{tyname}-best', [ s.sel(kmer=kmers) for (s,c) in scs ])
            # worst
            kmers = list(reduce(set.union, [set(c[:low]) for (_,c) in scs]))
            plotForest(f'{w}-{tyname}-worst', [ s.sel(kmer=kmers) for (s,c) in scs ])
        # then run plot on unification
        #kmers = list(set(kmers))
        #for w in ["scale", "mad"]:
        #    scs = []
        #    for t in ts:
        #        means = abs(t.posterior[w].mean(("chain","draw")))
        #        z = means / 1 # t.posterior["scale"].std(("chain","draw"))
        #        s = t.posterior[w].sortby(z)
        #        c = z.sortby(z).coords['kmer'].values
        #        scs.append((s,c))
        #    plotForest(f'{w}-{tyname}-bestqq', [ s.sel(kmer=kmers) for (s,c) in scs ])
    #go (-4,3,"sep",ts)
    go (-10,10,"join",[az.concat(ts, dim="draw")])

class MyLabeller(azl.BaseLabeller):
    def make_label_flat(self, var_name: str, sel: dict, isel: dict):
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{sel_str}"

def plotForest(name, traces):
    _, _, n = traces[0].shape
    plt.rcParams["font.family"] = "monospace"
    ySize = min(256, n)
    fig, ax = plt.subplots(figsize=(6, ySize))
    ax.set_facecolor('white')
    ax.set_xlabel('Weight', fontsize=fontsz)
    ax.set_ylabel('K-mers', fontsize=fontsz)
    legend = ax.get_legend()
    #if legend is not None:
    #    for text in legend.get_texts():
    #        if text is not None:
    #            text.set(fontsize=fontsz)
    plt.grid(c='grey')
    mn = None
    if (len(traces)>1):
        mn = [ f'cross-{i}' for i in list(range(1,len(traces)+1))]
    labeller = MyLabeller()
    az.plot_forest(traces, var_names=['~p'], figsize=(6,ySize), ax=ax
                  ,model_names = mn, textsize=fontsz, markersize=14, linewidth=4,labeller=labeller)
    plt.savefig(f'{name}.png')
    plt.savefig(f'{name}.pdf')
    plt.close()

#    positions = pd.DataFrame(
#        data=0, columns=["A", "C", "G", "T"], index=list(range(1, int(kmer)+1)))
#    posData = abs(trace.posterior["scale"].mean(("chain", "draw")))
#    posData = posData
#    print(posData)
#    for cell in posData:
#        nucs = cell["kmer"].item()
#        v = float(cell.values)
#        for i, n in enumerate(nucs):
#            positions.at[i+1, n] = positions.at[i+1, n] + \
#                (v / 4**(float(kmer)-1))
#    log.info(positions)
#    sb.heatmap(positions, annot=True, fmt=".2f", annot_kws={"size": 20})
#    plt.savefig(f'{fnamepfx}-positionimportance.png')
#    plt.savefig(f'{fnamepfx}-positionimportance.pdf')
#    plt.close()
def posImportance(kmer,fs):
    ts = [ az.from_netcdf(join(f,f'5-adagrad-trace.netcdf')) for f in fs ]
    positions = []
    # go over all traces, collect the kmers and scores, append to positions
    for it,t in enumerate(ts):
        posData=abs(t.posterior['scale'].mean(('chain','draw')))
        for p in posData:
            ns = p['kmer'].item()
            v = float(p.values)
            for i,n in enumerate(ns):
                positions.append({'Nucleotide': n, 'Position': i, 'cross': it, 'v': v})
    # now create mean and stddev for each
    df = pd.DataFrame(positions)
    # mean over chains and draws
    npc = df.groupby(['Nucleotide','Position','cross']).median()
    # final table
    npmean = npc.groupby(['Nucleotide','Position']).mean()
    npstd = npc.groupby(['Nucleotide','Position']).std()
    npmean = npmean.reset_index()
    npstd = npstd.reset_index()
    print(npmean)
    npmean = npmean.pivot(index='Position', columns='Nucleotide', values='v').to_numpy()
    print(npmean)
    npstd = npstd.pivot(index='Position', columns='Nucleotide', values='v').to_numpy()
    print(npstd)
    annot = np.char.add( np.vectorize(lambda v: f'{v:.2f}')(npmean)
                  , np.vectorize(lambda v: f'\n(±{v:.2f})')(npstd))
    sb.heatmap(npmean, annot=annot, fmt='', annot_kws={"size":15}, xticklabels=['A','C','G','T'], yticklabels=['-2','-1','0','+1','+2'])
    plt.savefig(f'positionimportance.png')
    plt.savefig(f'positionimportance.pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdirs', nargs='+',
                        action='append', help='directories with input files')
    parser.add_argument('--kmer')
    parser.add_argument('--zerolabel')
    parser.add_argument('--onelabel')
    parser.add_argument('--withmean', action='store_true', default=False)
    parser.add_argument('--withstddev', action='store_true', default=False)
    parser.add_argument('--withlines', action='store_true', default=False)
    parser.add_argument('--fdr')
    args = parser.parse_args()
    ids = [f for fs in args.inputdirs for f in fs]
    print(ids)
    plotFDR(ids, args.kmer, args.withmean, args.withstddev, args.withlines, args.fdr)
    plotErrorResponse(ids, args.kmer, args.zerolabel, args.onelabel, args.withmean, args.withstddev, args.withlines)
    plotForests(ids)
    posImportance(args.kmer,ids)

if __name__ == "__main__":
    main()

from random import shuffle
from scipy import stats
import pytensor
import pytensor.tensor as at
import arviz as az
import logging as log
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import numpy as np
import pymc as pm
import scipy
import xarray as xr
import random
import seaborn as sb
import pandas as pd
import os.path
import sys
import csv
import arviz.labels as azl

import pymc.sampling.jax

import Stats
from Construct import SummaryStats
import Kmer

# Always use the same seed for reproducability. (We might want to make this a cmdline argument)

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 1000
fontsz = 18  # fontsize
titlesz = 26  # fontsize for title
linew = 2  # line width of important lines
cOrange = '#d95f02'
cBlue = '#7570b3'
cGreen = '#1b9e77'


"""
"""


def genKcoords(k):
    assert (k > 0)
    if k == 1:
        return ['A', 'C', 'G', 'T']
    else:
        s1 = genKcoords(1)
        sk = genKcoords(k-1)
        return [s+ss for s in s1 for ss in sk]


"""
Builds the model. Building is a bit abstracted to simplify handing over mini batches for
optimization.
"""


def buildModel(coords, preMedian, medianZ, madZbc, meanDwellbc, observed, kmer, totalSz, batchSz=1000, dwellTimes=True):
    with pm.Model(coords=coords) as model:
        log.info(f'preMedian shape: {preMedian.shape}')
        log.info(f'medianZ shape: {medianZ.shape}')
        log.info(f'madZbc shape: {madZbc.shape}')
        log.info(f'meanDwellbc shape: {meanDwellbc.shape}')
        log.info(f'observed shape: {observed.shape}')

        pScale = pm.Beta('preScale', 0.5, 0.5) # this is a scalar
        kScale = pm.Normal('scale', 0, 1, dims='kmer')
        mScale = pm.Normal('mad', 0, 1, dims='kmer')
        intercept = pm.Normal('intercept', 0, 10)
        dwellScale = None
        if dwellTimes:
            dwellScale = pm.Normal('dwell', 0, 1, dims='kmer')
        log.info(f'pScale shape: {pScale.shape}')
        log.info(f'kScale shape: {kScale.shape}')
        log.info(f'mScale shape: {mScale.shape}')
        log.info(f'intercept shape: {intercept.shape}')

        rowSum = pm.math.dot(medianZ, kScale) # medianZ-preChange, does not improve predictions, hence left out for now
        rowSum += pm.math.dot(madZbc, mScale)
        # BUG Use switch to enable / disable dwell times (not a "bug", rather convenience since numbers show that dwell times are not helpful -- for D2O!)
        if dwellTimes:
            rowSum += pm.math.dot(meanDwellbc, dwellScale)
        predpcnt = pm.Deterministic('p', pm.math.invlogit(intercept + rowSum * pScale))
        log.info(f'sum shapes: {rowSum.shape} {predpcnt.shape}')

        # obs = pm.Normal("obs", mu=predpcnt, sigma=err, observed=pcnt)
        # BUG claim of imputed values for observed. Why?
        obs = pm.Bernoulli("obs", p=predpcnt,
                           observed=observed, total_size=totalSz)
        log.info(f'obs shape: {obs.shape}')
        log.info(f'obs: {obs}')
    return model


def buildTensorVars(preMedian, medianZ, madZbc, obs, dwellMeanbc):
    nmPreMedian = pytensor.shared(preMedian)
    nmMedianZ = pytensor.shared(medianZ)
    nmMadZbc = pytensor.shared(madZbc)
    nmObs = pytensor.shared(obs)
    nmDwellMeanbc = pytensor.shared(dwellMeanbc)
    return nmPreMedian, nmMedianZ, nmMadZbc, nmObs, nmDwellMeanbc

# The holy model:
# - logistic regression on 0, 30, 100 % deuterium; i.e 0; 0.3; 1.0
# - individual data can be pulled up or down by the pre-median calibration
# - the k1med and friends provide scaling on individual data points
# - deuterium ~ logistic (k1med * (x - premedian * PMS))
#
# - possibly set sigma to mad

# TODO consider normalization
# TODO make sure to select the correct "rel"s


def runModel(zeroRel, oneRel, outputDir, kmer, constructs, train=True, posteriorpredictive=True, priorpredictive=True, maxsamples=None, sampler="jax", batchSz=1000, dwellTimes=True):

    fnamepfx = os.path.join(outputDir, f'{kmer}-{sampler}')

    print(constructs[0].df)

    # construct the "df" necessary for the model to continue. this is probably wasteful, but allows us to quickly continue from here on.
    df = pd.concat([c.df for c in constructs])
    print(df)
    print(float(zeroRel))


    df['p'].loc[df['p'] == float(zeroRel)] = 0
    df['p'].loc[df['p'] == float(oneRel)] = 1
    df['p']
    print('ZERO', df[df['p'] == 0])
    print('ONE', df[df['p'] == 1])
    print('LEN', len(df))

    # prepare subsampling
    rels = df['p'].value_counts()
    log.info(f'p: {rels}')
    samplecount = int(min(rels))
    if maxsamples is not None:
        samplecount = min(samplecount, int(maxsamples))
    sampledreads = []
    for i in rels.index:
        cands = list(set(df[df['p'] == i].index))
        random.shuffle(cands)
        sampledreads.extend(cands[0:samplecount])
    log.info(f'subsampled {samplecount} reads for each d2o level')
    df = df[df.index.isin(sampledreads)]

    # concatenate all 'madZ' values, perform boxcox to get the lambda
    madzs = np.concatenate(df['madZ'])
    madzs = np.delete(madzs, np.where(madzs == 0))
    _ , lmbdMadZ = scipy.stats.boxcox(madzs)
    log.info(f'BoxCox Lambda: {lmbdMadZ}')
    df['madZbc'] = df['madZ'].apply(lambda x: scipy.stats.boxcox(x, lmbda = lmbdMadZ))

    # concatenate all 'dwellMean' values, perform boxcox again
    dwellmeans = np.concatenate(df['dwellMean'])
    dwellmeans = np.delete(dwellmeans, np.where(dwellmeans==0))
    _, lmbdDwellMeans = scipy.stats.boxcox(dwellmeans)
    df['dwellMeanbc'] = df['dwellMean'].apply(lambda x: scipy.stats.boxcox(x, lmbda = lmbdDwellMeans))
    df['dwellMeanbc'] = df['dwellMeanbc'].apply(lambda x: np.nan_to_num(x, nan=0, posinf=0, neginf=0))

    # determine "nan" reads!
    #nans = np.isnan(df['rel'].to_numpy())
    #print('LEN', len(nans) / (4**int(kmer)))
    #print('nans', len(df[nans]))

    # transform the column into the correct matrix form
    #medianZ = df['medianZ'].to_xarray()
    medianZ = np.array(list(df['medianZ']))
    #madZbc = df['madZbc'].to_xarray()
    madZbc = np.array(list(df['madZbc']))
    #rel = df['p'].to_xarray()
    rel = np.array(list(df['p']))
    #preMedian = df['pfxZ'].to_xarray()
    preMedian = np.array(list(df['pfxZ']))
    #relTotalSize = rel.shape
    dwellMeanbc = np.array(list(df['dwellMeanbc']))

    # TODO hints on how to implement minibatches, which will require creating a single, huge matrix,
    # then splitting again
    # print(medianZ)
    # print(medianZ.to_numpy())
    # zzz = pm.Minibatch(medianZ.to_numpy(), batch_size=128)
    # print(zzz.eval().shape)

    # prepare coords
    coords = {'kmer': Kmer.gen(int(kmer))
              }

    nmPreMedian, nmMedianZ, nmMadZbc, nmRel, nmDwellMeanbc = buildTensorVars(
        preMedian, medianZ, madZbc, rel, dwellMeanbc)

    # TODO profile model (especially for k=5)

    # prior predictive checks needs to be written down still
    # if priorpredictive:
    #  pass
    #  #with model:
    #  #  log.info('running prior predictive model')
    #  #  trace = pm.sample_prior_predictive()
    #  #  _, ax = plt.subplots()
    #  #  x = xr.DataArray(np.linspace(-5,5,10), dims=["plot_dim"])
    #  #  prior = trace.prior
    #  #  ax.plot(x, x)
    #  #  plt.savefig(f'{kmer}-prior-predictive.jpeg')
    #  #  plt.close()

    trace = az.InferenceData()
    if train:
        # Switch between different options, jax or pymc nuts with advi
        log.info('training model RUN')
        model = None
        mf = None  # mean field
        trace = None
        match sampler:
            case "adagrad":
                mbPreMedian, mbMedianZ, mbMadZbc, mbRel, mbDwellMeanbc = pm.Minibatch(
                    nmPreMedian, nmMedianZ, nmMadZbc, nmRel, nmDwellMeanbc, batch_size=batchSz)
                model = buildModel(coords, mbPreMedian,
                                   mbMedianZ, mbMadZbc, mbDwellMeanbc, mbRel, kmer, rel.shape, dwellTimes)
                # run the model with mini batches
                with model:
                    mf: pm.MeanField = pm.fit(
                        n=50000, obj_optimizer=pm.adagrad_window(learning_rate=1e-2))
                    trace = mf.sample(draws=5000)
            #case "advi":
            #    mbPreMedian, mbMedianZ, mbMadZbc, mbRel = pm.Minibatch(
            #        nmPreMedian, nmMedianZ, nmMadZbc, nmRel, batch_size=batchSz)
            #    model = buildModel(coords, mbPreMedian,
            #                       mbMedianZ, mbMadZbc, mbRel, kmer, rel.shape)
            #    with model:
            #        mf = pm.fit(n=50000, method="advi")
            #        trace = mf.sample(draws=5000)
            #case "jax":
            #    model = buildModel(coords, preMedian,
            #                       medianZ, madZbc, rel, kmer)
            #    with model:
            #        trace = pymc.sampling.jax.sample_numpyro_nuts(
            #            draws=1000, tune=1000, chains=2, postprocessing_backend='cpu')
            #case "nuts":
            #    trace = pm.sample(1000, return_inferencedata=True,
            #                      tune=1000, chains=2, cores=2)
            case "advi-nuts":
                model = buildModel(coords, preMedian, medianZ,
                                   madZbc, dwellMeanbc, rel, kmer, rel.shape)
                with model:
                    trace = pm.sample(1000, return_inferencedata=True,
                                      tune=1000, chains=2, cores=2, init="advi+adapt_diag")
        assert trace is not None
        trace.to_netcdf(f'{fnamepfx}-trace.netcdf')
        # TODO pickle the trace
        log.info('training model DONE')
    else:
        # TODO possibly load model
        log.info('training model LOAD')
        try:
            trace = trace.from_netcdf(f'{fnamepfx}-trace.netcdf')
        except OSError:
            log.info(f'Missing netcdf file: {fnamepfx}-trace.netcdf not found')
            sys.exit(1)
        log.info('training model DONE')
        pass
    #
    # Determine the importance of positions / nucleotides
    #
    log.info('position importance')
    positions = pd.DataFrame(
        data=0, columns=["A", "C", "G", "T"], index=list(range(1, int(kmer)+1)))
    posData = abs(trace.posterior["scale"].mean(("chain", "draw")))
    posData = posData
    print(posData)
    for cell in posData:
        nucs = cell["kmer"].item()
        v = float(cell.values)
        for i, n in enumerate(nucs):
            positions.at[i+1, n] = positions.at[i+1, n] + \
                (v / 4**(float(kmer)-1))
    log.info(positions)
    sb.heatmap(positions, annot=True, fmt=".2f", annot_kws={"size": 20})
    plt.savefig(f'{fnamepfx}-positionimportance.png')
    plt.savefig(f'{fnamepfx}-positionimportance.pdf')
    plt.close()
    # create plot(s); later guard via switch
    if True:
        ySize = min(256, 4**float(kmer))
        plotMcmcTrace(fnamepfx, kmer, trace)
        print(az.summary(trace, var_names=['~p'], round_to=2))
        # BUG need to plot forest without any sorting ...
        ##
        # scale stuff
        ##
        scaleMeans = abs(trace.posterior["scale"].mean(("chain", "draw")))
        scaleZ = scaleMeans / trace.posterior["scale"].std(("chain", "draw"))
        sortedScaleTrace = trace.posterior["scale"].sortby(scaleZ)
        scaleCoords = scaleZ.sortby(scaleZ).coords['kmer'].values
        # best 10 scale values
        plotForest(fnamepfx, 'zsortedforest-scale-worst', kmer,
                   sortedScaleTrace.sel(kmer=scaleCoords[:12]))
        plotForest(fnamepfx, 'zsortedforest-scale-best', kmer,
                   sortedScaleTrace.sel(kmer=scaleCoords[-12:]))
        ##
        # mad stuff
        ##
        madMeans = abs(trace.posterior["mad"].mean(("chain", "draw")))
        madZ = madMeans / trace.posterior["mad"].std(("chain", "draw"))
        sortedMadTrace = trace.posterior["mad"].sortby(madZ)
        madCoords = madZ.sortby(madZ).coords['kmer'].values
        plotForest(fnamepfx, 'zsortedforest-mad-worst', kmer,
                   sortedMadTrace.sel(kmer=madCoords[:12]))
        plotForest(fnamepfx, 'zsortedforest-mad-best', kmer,
                   sortedMadTrace.sel(kmer=madCoords[-12:]))
        ##
        # dwell time
        ##
        madMeans = abs(trace.posterior['dwell'].mean(("chain", "draw")))
        madZ = madMeans / trace.posterior['dwell'].std(("chain", "draw"))
        sortedMadTrace = trace.posterior['dwell'].sortby(madZ)
        madCoords = madZ.sortby(madZ).coords['kmer'].values
        plotForest(fnamepfx, 'zsortedforest-dwell-worst', kmer,
                   sortedMadTrace.sel(kmer=madCoords[:12]))
        plotForest(fnamepfx, 'zsortedforest-dwell-best', kmer,
                   sortedMadTrace.sel(kmer=madCoords[-12:]))

    # plot the posterior, should be quite fast
    # TODO only plots subset of figures, if there are too many subfigures
    log.info(f'plotting posterior distributions')
    plotPosterior(fnamepfx, trace)

    if posteriorpredictive:
        # rebuild the model with full data!
        # model = buildModel(coords, preMedian, medianZ, madZbc, rel, kmer, rel.shape)
        model = buildModel(coords, nmPreMedian, nmMedianZ,
                           nmMadZbc, nmDwellMeanbc, nmRel, kmer, rel.shape)
        with model:
            log.info('posterior predictive run START')
            assert trace is not None
            # TODO
            # Normally, we should now go and set new data via
            # pm.set_data({"pred": out-of-sample-data})
            # but we can pickle the trace, then reload with new data
            thinnedTrace = trace.sel(draw=slice(None, None, 5))
            trace = pm.sample_posterior_predictive(thinnedTrace, var_names=[
                                                   'p', 'obs'], return_inferencedata=True, extend_inferencedata=True, predictions=True)
            log.info('posterior predicte run DONE')
            print(trace)
            # important: contains "p"
            mpreds = trace['predictions']
            log.info(mpreds)
            mppmean = mpreds['p'].mean(axis=(0, 1))
            mppmean = mppmean.rename({'p_dim_2': 'read'})
            mppstd = mpreds['p'].std(axis=(0, 1))
            mppstd = mppstd.rename({'p_dim_2': 'read'})
            print('mppmean', mppmean)
            print('mppmean.coords', mppmean.coords)
            print('rel', rel)
            obs = xr.DataArray(data=rel, coords=mppmean.coords)
            print('obs', obs)
            # inplace sorting of the results, keeps means and variances associated
            mppmean = mppmean.sortby(mppmean)
            mppstd = mppstd.sortby(mppmean)
            obs = obs.sortby(mppmean)
            # plotPosteriorPredictiveError(fnamepfx, mppmean, obs)
            plotErrorResponse(fnamepfx, zeroRel, oneRel, mppmean, obs)
            falseDiscoveryRate(fnamepfx, mppmean, obs)


# Plots the model response error. For each sample, presence of isotopes is
# predicted within [0..1]. The error is the distance to the true class. In
# addition, we plot a horizontal line at 0.5. All points below are predicted
# correctly, all points above incorrectly. Two vertical lines show the fraction
# until which the error rate is below 0.5.

# BUG Check for bugs, including NANs, INFs, etc. This probably leads to the weird class problems here!

def plotErrorResponse(fnamepfx, zeroRel, oneRel, mppmean, obs):
    print(mppmean)
    print(obs)
    aom = (mppmean-obs)
    print(aom)
    p0 = aom.where(lambda x: x >= 0, drop=True)
    p0 = p0.sortby(p0)
    p1 = abs(aom.where(lambda x: x <= 0, drop=True))
    p1 = p1.sortby(p1)
    minp0p1 = min(len(p0),len(p1))
    cands = list(range(0,max(len(p0),len(p1))-1))
    random.shuffle(cands)
    if (len(p0)>len(p1)):
        cs = cands[:len(p1)]
        cs.sort()
        p0 = p0[cs]
    if (len(p1)>len(p0)):
        cs = cands[:len(p0)]
        cs.sort()
        p1 = p1[cs]
    # TODO fixup lengths via sampling
    p0good = len(p0.where(lambda x: x < 0.5, drop=True))
    p1good = len(p1.where(lambda x: x < 0.5, drop=True))
    _, ax = plt.subplots(figsize=(6, 6))
    print(len(p0),len(p1))
    # BUG check on NaN data, that prevents this assertion. Sample from the larger vector then to allow printing. Assert failure is likely with bad data, say from Carbon.
    assert len(p0) == len(p1)
    if (len(p0) != len(p1)):
        # TODO sample longer one...
        pass
    df = pd.DataFrame({'p0': p0, 'p1': p1})
    df.to_csv(f'{fnamepfx}-response.csv', index=False)
    ax.set_facecolor('white')
    plt.grid(c='grey')
    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=linew)
    ax.plot(p0, color=cOrange,
            label=f'{float(zeroRel) * 100}%', linewidth=linew)
    plt.axvline(x=p0good, color=cOrange, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p0good / max(1,len(p0)):.2f}',
                 xy=(p0good, 0.6), color=cOrange, fontsize=fontsz)
    ax.plot(p1, color=cBlue, label=f'{float(oneRel) * 100}%', linewidth=linew)
    plt.axvline(x=p1good, color=cBlue, linestyle='solid', linewidth=linew)
    plt.annotate(f'{p1good / max(1,len(p1)):.2f}',
                 xy=(p1good, 0.4), color=cBlue, fontsize=fontsz)
    # horizontal line at error 0.5
    ax.set_xlabel('Reads (ordered by distance)', fontsize=fontsz)
    ax.set_ylabel('Distance to true class (lower is better)', fontsize=fontsz)
    #ax.set_title('Error response', fontsize=titlesz)
    # ax.legend(frameon=True, framealpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=1.0,
              loc='upper left', bbox_to_anchor=(0.1, 0.9))
    # TODO vertical line that is annotated with percentage "good"
    plt.savefig(f'{fnamepfx}-model-error.png')
    plt.savefig(f'{fnamepfx}-model-error.pdf')
    plt.close()


def plotPosteriorPredictiveError(fname, mppmean, obs):
    _, ax = plt.subplots(figsize=(12, 6))
    # mean with +- stddev
    ax.plot(mppmean, color=cBlue)
    ax.plot(mppmean + mppstd, color=cBlue)
    ax.plot(mppmean - mppstd, color=cBlue)
    ax.plot(obs, '.')
    # actual
    ax.set_xlabel('Samples (sorted by p)')
    ax.set_ylabel('p (Predicted D2O)')
    ax.set_title('Posterior Predictive Error (±σ)')
    ax.legend(frameon=True, framealpha=0.5)
    plt.savefig(f'{fnamepfx}-poos.png')
    plt.savefig(f'{fnamepfx}-poos.pdf')
    plt.close()

# Plots the MCMC traces and overlay of all posteriors. Will already give a hint if some parameters are more important than others.


def plotMcmcTrace(fnamepfx, kmer, trace):
    xSize, ySize = 12, 12
    # fig, ax = plt.subplots(figsize=(xSize,ySize))
    axes = az.plot_trace(trace, compact=True, combined=True,
                         var_names=['~p'], figsize=(xSize, ySize))
    for ax in axes.flatten():
        plt.grid(c='grey')
        ax.set_facecolor('white')
    plt.savefig(f'{fnamepfx}-trace.png')
    plt.savefig(f'{fnamepfx}-trace.pdf')
    plt.close()

# Forest plot of parameters.

class MyLabeller(azl.BaseLabeller):
    def make_label_flat(self, var_name: str, sel: dict, isel: dict):
        var_name_str = self.var_name_to_str(var_name)
        sel_str = self.sel_to_str(sel, isel)
        if not sel_str:
            return "" if var_name_str is None else var_name_str
        if var_name_str is None:
            return sel_str
        return f"{sel_str}"

def plotForest(fnamepfx, fnamessfx, kmer, trace):
    _, _, n = trace.shape
    plt.rcParams["font.family"] = "monospace"
    ySize = min(256, n/2)
    fig, ax = plt.subplots(figsize=(6, ySize))
    ax.set_facecolor('white')
    plt.grid(c='grey')
    #labeller = azl.MapLabeller(var_name_map={"scale": "", "mad": ""})
    labeller = MyLabeller()
    az.plot_forest(trace, var_names=['~p'], figsize=(6, ySize), ax=ax, labeller=labeller)
    # legend = ax.get_legend()
    # for text in legend.get_texts():
    #    text.set(fontfamily='monospace')
    plt.savefig(f'{fnamepfx}-{fnamessfx}.png')
    plt.savefig(f'{fnamepfx}-{fnamessfx}.pdf')
    plt.close()


def plotPosterior(fnamepfx, trace):
    xSize, ySize = 12, 6
    axes = az.plot_posterior(
        trace, var_names=['intercept'], figsize=(xSize, ySize)) # 'preScale'
    plt.grid(c='grey')
    axes.set_facecolor('white')
    #for ax in axes.flatten():
    #    plt.grid(c='grey')
    #    ax.set_facecolor('white')
    plt.savefig(f'{fnamepfx}-posterior.png')
    plt.savefig(f'{fnamepfx}-posterior.pdf')
    plt.close()


"""
Calculate the false discovery rate at certain levels and plot
"""

# TODO Consider using not mppmean, but the whole set of parallel runs. Then provide +- 1 sigma around the mean as well.


def falseDiscoveryRate(fnamepfx, mppmean, obs):
    # Only choose elements *at most* this far away from 0.5; 0.1, for example, accept <=0.1 and >=0.9
    rsStepSz = 0.0001
    rs = np.arange(0 + rsStepSz, 0.5 + rsStepSz, rsStepSz)
    ys = []
    ns = []  # relative fraction of reads within the constraint
    for r in rs:
        cond = np.logical_or(mppmean <= r, mppmean >= 1-r)
        ms = mppmean[cond]
        os = obs[cond]
        cs = abs(ms-os)
        fs = cs[cs > 0.5]
        ys = np.append(ys, len(fs) / max(1, len(cs)))
        ns = np.append(ns, len(ms) / len(mppmean))
    # this data is also useful for cross validation plots
    df = pd.DataFrame({'cutoff': rs, 'fdr': ys, 'relreads': ns})
    df.to_csv(f'{fnamepfx}-fdr.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.grid(None)
    ax1.set_facecolor('white')
    ax1.plot(rs, ys, color='black', label=f'FDR')
    ax1.plot(rs, ns, color='blue', label='% reads')
    ax1.set_ylabel('% reads', fontsize=fontsz)
    ax1.set_xlabel('Cutoff', fontsize=fontsz)
    #ax1.set_title('False discovery rate', fontsize=titlesz)
    plt.grid(c='grey')
    ax2 = ax1.twinx()
    ax2.set_ylabel('FDR', fontsize=fontsz)
    fig.legend(frameon=True, facecolor='white', framealpha=1.0,
               loc='lower right', bbox_to_anchor=(0.85, 0.15))
    plt.savefig(f'{fnamepfx}-fdr.png')
    plt.savefig(f'{fnamepfx}-fdr.pdf')
    plt.close()

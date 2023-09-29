
from random import shuffle
from scipy import stats
import pytensor
import pytensor.tensor as at
import arviz as az
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy
import xarray as xr
import random
import seaborn as sb
import pandas as pd
import os.path
import sys

import pymc.sampling.jax

import Stats
from Construct import SummaryStats
import Kmer

# Always use the same seed for reproducability. (We might want to make this a cmdline argument)

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 1000
#aesara.config.profile = True
cOrange = '#d95f02'
cBlue   = '#7570b3'
cGreen  = '#1b9e77'


"""
"""
def genKcoords (k):
  assert (k>0)
  if k==1:
    return [ 'A', 'C', 'G', 'T' ]
  else:
    s1 = genKcoords(1)
    sk = genKcoords(k-1)
    return [ s+ss for s in s1 for ss in sk ]

"""
Builds the model. Building is a bit abstracted to simplify handing over mini batches for
optimization.
"""
def buildModel(coords, preMedian, medianZ, madZbc, observed, kmer, totalSz, batchSz = 1000):
  with pm.Model(coords = coords) as model:
    # data we want to be able to swap for posterior predictive
    # access via get_value() / set_value()
    #preMedian = pm.MutableData("preMedian", preMedian)
    #medianZ  = pm.MutableData("medianZ", medianZ)
    #madZbc   = pm.MutableData('madZbc', madZbc)
    #log.info(f'preMedian data shape: {preMedian.get_value().shape}')
    #log.info(f'medianZ data shape: {medianZ.get_value().shape}')
    #log.info(f'madZbc data shape: {madZbc.get_value().shape}')

    pScale    = pm.Beta('preScale', 0.5, 0.5)
    kScale    = pm.Normal('scale', 0, 1, dims='kmer')
    mScale    = pm.Normal('mad', 0, 1, dims='kmer')
    intercept = pm.Normal('intercept', 0, 10)
    log.info(f'pScale shape: {pScale.shape}')
    log.info(f'kScale shape: {kScale.shape}')
    log.info(f'mScale shape: {mScale.shape}')
    log.info(f'intercept shape: {intercept.shape}')

    #rowSum    =  pm.math.dot(medianZ, kScale)
    rowSum    =  pm.math.dot(medianZ - pScale * preMedian, kScale)
    rowSum    += pm.math.dot(madZbc, mScale)
    predpcnt  = pm.Deterministic('p', pm.math.invlogit(intercept + rowSum))
    log.info(f'sum shapes: {rowSum.shape} {predpcnt.shape}')


    #obs = pm.Normal("obs", mu=predpcnt, sigma=err, observed=pcnt)
    # BUG claim of imputed values for observed. Why?
    obs = pm.Bernoulli("obs", p=predpcnt, observed=observed, total_size = totalSz)
    log.info(f'obs shape: {obs.shape}')
    log.info(f'obs: {obs}')
  return model

def buildTensorVars(preMedian, medianZ, madZbc, obs):
  nmPreMedian = preMedian.to_numpy()
  nmMedianZ = medianZ.to_numpy()
  nmMadZbc = madZbc.to_numpy()
  nmObs = obs.to_numpy()
  return nmPreMedian, nmMedianZ, nmMadZbc, nmObs

# The holy model:
# - logistic regression on 0, 30, 100 % deuterium; i.e 0; 0.3; 1.0
# - individual data can be pulled up or down by the pre-median calibration
# - the k1med and friends provide scaling on individual data points
# - deuterium ~ logistic (k1med * (x - premedian * PMS))
#
# - possibly set sigma to mad

# TODO consider normalization
# TODO make sure to select the correct "rel"s

def runModel(zeroRel, oneRel, outputDir, kmer, df, train = True, posteriorpredictive = True, priorpredictive = True, maxsamples = None, sampler = "jax", batchSz=1000):

  fnamepfx = os.path.join(outputDir, f'{kmer}-{sampler}')

  # Perform set selection
  df = df[(df['rel']==float(zeroRel)) | (df['rel']==float(oneRel))]
  print(df)
  df = df.groupby('read').filter(lambda x: len(x) == 4**int(kmer))
  print(df)
  df['rel'].loc[df['rel']==float(zeroRel)] = 0
  df['rel'].loc[df['rel']==float(oneRel)] = 1
  df['rel']
  print('ZERO', df[df['rel']==0])
  print('ONE', df[df['rel']==1])
  print('LEN', len(df) / (4**int(kmer)))

  # prepare subsampling
  rels = df['rel'].value_counts()
  samplecount = int(min(rels) / (4**int(kmer)))
  if maxsamples is not None:
    samplecount = min(samplecount, int(maxsamples))
  sampledreads = []
  for i in rels.index:
    cands = list(set(df[df['rel']==i].droplevel('k').index))
    random.shuffle(cands)
    sampledreads.extend(cands[0:samplecount])
  log.info(f'subsampled {samplecount} reads for each d2o level')
  df = df[df.droplevel('k').index.isin(sampledreads)]

  # The "madZ" values are all positive. We apply a Box-Cox transformation here
  meanmadz = df[df['madZ']>0]['madZ'].mean()
  # TODO Try using .loc[row_indexer,col_indexer] = value instead
  df['madZ'] = df['madZ'].apply(lambda x: x if x > 0 else meanmadz)
  mads, lmbd = scipy.stats.boxcox(df['madZ'])
  df = df.assign(madZbc = mads)
  print('LEN', len(df) / (4**int(kmer)))
  #
  # NAN tests
  #print(len(df))
  #df = df[~np.isnan(df).any(axis=1)]
  #print(len(df))
  #df = df[~np.isinf(df).any(axis=1)]
  #print(len(df))

  # determine "nan" reads!
  nans = np.isnan(df['rel'].to_numpy())
  print('LEN', len(nans) / (4**int(kmer)))
  print('nans', len(df[nans]))

  # transform the column into the correct matrix form
  medianZ = df['medianZ'].to_xarray()
  madZbc = df['madZbc'].to_xarray()
  rel = df['rel'].to_xarray()[:,0]
  rel = rel.drop_vars('k')
  preMedian = df['pfxZ'].to_xarray()
  relTotalSize = rel.shape

  # TODO hints on how to implement minibatches, which will require creating a single, huge matrix,
  # then splitting again
  #print(medianZ)
  #print(medianZ.to_numpy())
  #zzz = pm.Minibatch(medianZ.to_numpy(), batch_size=128)
  #print(zzz.eval().shape)

  # prepare coords
  coords = { 'kmer': Kmer.gen(int(kmer))
           }

  nmPreMedian, nmMedianZ, nmMadZbc, nmRel = buildTensorVars(preMedian, medianZ, madZbc, rel)


  # TODO profile model (especially for k=5)

  # prior predictive checks needs to be written down still
  #if priorpredictive:
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
    mf = None # mean field
    trace = None
    match sampler:
      case "adagrad":
        mbPreMedian, mbMedianZ, mbMadZbc, mbRel = pm.Minibatch(nmPreMedian, nmMedianZ, nmMadZbc, nmRel, batch_size=batchSz)
        model = buildModel(coords, mbPreMedian, mbMedianZ, mbMadZbc, mbRel, kmer, rel.shape)
        # run the model with mini batches
        with model:
          mf : pm.MeanField = pm.fit(n=50000, obj_optimizer=pm.adagrad_window(learning_rate=1e-2))
          trace = mf.sample(draws=5000)
      case "advi":
        mbPreMedian, mbMedianZ, mbMadZbc, mbRel = pm.Minibatch(nmPreMedian, nmMedianZ, nmMadZbc, nmRel, batch_size=batchSz)
        model = buildModel(coords, mbPreMedian, mbMedianZ, mbMadZbc, mbRel, kmer, rel.shape)
        with model:
          mf = pm.fit(n=50000, method="advi")
          trace = mf.sample(draws=5000)
      case "jax":
        model = buildModel(coords, preMedian, medianZ, madZbc, rel, kmer)
        with model:
          trace = pymc.sampling.jax.sample_numpyro_nuts(draws=1000, tune=1000, chains=2, postprocessing_backend='cpu')
      case "nuts":
        trace = pm.sample(1000, return_inferencedata=True, tune=1000, chains=2, cores=2)
      case "advi-nuts":
        model = buildModel(coords, nmPreMedian, nmMedianZ, nmMadZbc, nmRel, kmer, nmRel.shape)
        #model = buildModel(coords, preMedian, medianZ, madZbc, rel, kmer, rel.shape)
        with model:
          trace = pm.sample(1000, return_inferencedata=True, tune=1000, chains=2, cores=2, init="advi+adapt_diag")
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
  positions = pd.DataFrame(data=0, columns=["A","C","G","T"], index=list(range(1,int(kmer)+1)))
  posData = abs(trace.posterior["scale"].mean(("chain", "draw")))
  posData = posData
  print(posData)
  for cell in posData:
    nucs = cell["kmer"].item()
    v = float(cell.values)
    for i,n in enumerate(nucs):
      positions.at[i+1,n] = positions.at[i+1,n] + (v / 4**(float(kmer)-1))
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
    ## scale stuff
    ##
    scaleMeans = abs(trace.posterior["scale"].mean(("chain", "draw")))
    scaleZ = scaleMeans / trace.posterior["scale"].std(("chain","draw"))
    sortedScaleTrace = trace.posterior["scale"].sortby(scaleZ)
    scaleCoords = scaleZ.sortby(scaleZ).coords['kmer'].values
    # best 10 scale values
    plotForest(fnamepfx, 'zsortedforest-scale-worst', kmer, sortedScaleTrace.sel(kmer=scaleCoords[:12]))
    plotForest(fnamepfx, 'zsortedforest-scale-best', kmer, sortedScaleTrace.sel(kmer=scaleCoords[-12:]))
    ##
    ## mad stuff
    ##
    madMeans = abs(trace.posterior["mad"].mean(("chain", "draw")))
    madZ = madMeans / trace.posterior["mad"].std(("chain","draw"))
    sortedMadTrace = trace.posterior["mad"].sortby(madZ)
    madCoords = madZ.sortby(madZ).coords['kmer'].values
    plotForest(fnamepfx, 'zsortedforest-mad-worst', kmer, sortedMadTrace.sel(kmer=madCoords[:12]))
    plotForest(fnamepfx, 'zsortedforest-mad-best', kmer, sortedMadTrace.sel(kmer=madCoords[-12:]))
    #az.plot_forest(sortedMadTrace, var_names=['~p'], figsize=(6,ySize))
    #plt.savefig(f'{fnamepfx}-zsortedforest-mad.png')
    #plt.savefig(f'{fnamepfx}-zsortedforest-mad.pdf')
    #plt.close()
    #bang
    #print(az.summary(sortedScaleTrace, var_names=['~p'], round_to=2))
    #sortedScaleTrace = trace.posterior["scale"].sortby(scaleMeans)
    #az.plot_forest(sortedScaleTrace, var_names=['~p'], figsize=(6,ySize))
    #plt.savefig(f'{fnamepfx}-meansortedforest-scale.png')
    #plt.savefig(f'{fnamepfx}-meansortedforest-scale.pdf')
    #plt.close()
    #print(az.summary(sortedScaleTrace, var_names=['~p'], round_to=2))
    ## mad stuff
    #madMeans = abs(trace.posterior["mad"].mean(("chain", "draw")))
    #madZ = madMeans / trace.posterior["mad"].std(("chain","draw"))
    #sortedMadTrace = trace.posterior["mad"].sortby(madZ)
    #az.plot_forest(sortedMadTrace, var_names=['~p'], figsize=(6,ySize))
    #plt.savefig(f'{fnamepfx}-zsortedforest-mad.png')
    #plt.savefig(f'{fnamepfx}-zsortedforest-mad.pdf')
    #plt.close()
    #print(az.summary(sortedMadTrace, var_names=['~p'], round_to=2))
    #sortedMadTrace = trace.posterior["mad"].sortby(madMeans)
    #az.plot_forest(sortedMadTrace, var_names=['~p'], figsize=(6,ySize))
    #plt.savefig(f'{fnamepfx}-meansortedforest-mad.png')
    #plt.savefig(f'{fnamepfx}-meansortedforest-mad.pdf')
    #plt.close()
    #print(az.summary(sortedMadTrace, var_names=['~p'], round_to=2))

  # plot the posterior, should be quite fast
  # TODO only plots subset of figures, if there are too many subfigures
  log.info(f'plotting posterior distributions')
  plotPosterior (fnamepfx, trace)

  if posteriorpredictive:
    # rebuild the model with full data!
    #model = buildModel(coords, preMedian, medianZ, madZbc, rel, kmer, rel.shape)
    print(nmPreMedian.shape)
    print(nmMedianZ.shape)
    print(nmRel.shape)
    model = buildModel(coords, nmPreMedian, nmMedianZ, nmMadZbc, nmRel, kmer, nmRel.shape)
    with model:
      log.info('posterior predictive run START')
      assert trace is not None
      # TODO
      # Normally, we should now go and set new data via
      # pm.set_data({"pred": out-of-sample-data})
      # but we can pickle the trace, then reload with new data
      thinnedTrace = trace.sel(draw=slice(None,None,5))
      trace = pm.sample_posterior_predictive(thinnedTrace, var_names=['p', 'obs'], return_inferencedata=True, extend_inferencedata=True, predictions=True)
      log.info('posterior predicte run DONE')
      print(trace)
      # important: contains "p"
      mpreds = trace['predictions']
      log.info(mpreds)
      mppmean = mpreds['p'].mean(axis=(0,1))
      mppmean = mppmean.rename({'p_dim_2' : 'read'})
      mppstd = mpreds['p'].std(axis=(0,1))
      mppstd = mppstd.rename({'p_dim_2' : 'read'})
      print('mppmean', mppmean)
      print('mppmean.coords', mppmean.coords)
      print('rel', rel)
      obs = xr.DataArray(data=rel, coords=mppmean.coords)
      print('obs', obs)
      # inplace sorting of the results, keeps means and variances associated
      mppmean = mppmean.sortby(mppmean)
      mppstd  = mppstd.sortby(mppmean)
      obs     = obs.sortby(mppmean)
      #plotPosteriorPredictiveError(fnamepfx, mppmean, obs)
      plotErrorResponse(fnamepfx, zeroRel, oneRel, mppmean, obs)

# Plots the model response error. For each sample, presence of isotopes is predicted within [0..1]. The error is the distance to the true class. In addition, we plot a horizontal line at 0.5. All points below are predicted correctly, all points above incorrectly. Two vertical lines show the fraction until which the error rate is below 0.5.

def plotErrorResponse (fnamepfx, zeroRel, oneRel, mppmean, obs):
    print(mppmean)
    print(obs)
    aom = (mppmean-obs)
    print(aom)
    p0 = aom.where(lambda x: x >= 0, drop=True)
    p0 = p0.sortby(p0)
    p0good = len(p0.where(lambda x: x < 0.5, drop=True))
    p1 = abs(aom.where(lambda x: x <  0, drop=True))
    p1 = p1.sortby(p1)
    p1good = len(p1.where(lambda x: x < 0.5, drop=True))
    _, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')
    plt.grid(c='grey')
    plt.axhline(y=0.5, color='black', linestyle='-')
    ax.plot(p0, color=cOrange, label=f'{float(zeroRel) * 100}%')
    plt.axvline(x=p0good, color=cOrange, linestyle='solid')
    plt.annotate(f'{p0good / max(1,len(p0)):.2f}', xy=(p0good,0.6), color=cOrange)
    ax.plot(p1, color=cBlue, label=f'{float(oneRel) * 100}%')
    plt.axvline(x=p1good, color=cBlue, linestyle='dashed')
    plt.annotate(f'{p1good / max(1,len(p1)):.2f}', xy=(p1good,0.4), color=cBlue)
    # horizontal line at error 0.5
    ax.set_xlabel('Samples (ordered)')
    ax.set_ylabel('Distance to true class (less is better)')
    ax.set_title('Error response')
    ax.legend(fontsize=10, frameon=True, framealpha=0.5)
    # TODO vertical line that is annotated with percentage "good"
    plt.savefig(f'{fnamepfx}-model-error.png')
    plt.savefig(f'{fnamepfx}-model-error.pdf')
    plt.close()

def plotPosteriorPredictiveError (fname, mppmean, obs):
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
      ax.legend(fontsize=10, frameon=True, framealpha=0.5)
      plt.savefig(f'{fnamepfx}-poos.png')
      plt.savefig(f'{fnamepfx}-poos.pdf')
      plt.close()

# Plots the MCMC traces and overlay of all posteriors. Will already give a hint if some parameters are more important than others.
def plotMcmcTrace(fnamepfx, kmer, trace):
    xSize, ySize = 12, 12
    #fig, ax = plt.subplots(figsize=(xSize,ySize))
    axes = az.plot_trace(trace, compact=True, combined=True, var_names=['~p'],figsize=(xSize,ySize))
    for ax in axes.flatten():
        plt.grid(c='grey')
        ax.set_facecolor('white')
    plt.savefig(f'{fnamepfx}-trace.png')
    plt.savefig(f'{fnamepfx}-trace.pdf')
    plt.close()

# Forest plot of parameters. 
def plotForest(fnamepfx, fnamessfx, kmer, trace):
    _,_,n = trace.shape
    plt.rcParams["font.family"] = "monospace"
    ySize = min(256, n/2)
    fig, ax = plt.subplots(figsize=(6, ySize))
    ax.set_facecolor('white')
    plt.grid(c='grey')
    az.plot_forest(trace, var_names=['~p'], figsize=(6,ySize),ax=ax)
    #legend = ax.get_legend()
    #for text in legend.get_texts():
    #    text.set(fontfamily='monospace')
    plt.savefig(f'{fnamepfx}-{fnamessfx}.png')
    plt.savefig(f'{fnamepfx}-{fnamessfx}.pdf')
    plt.close()

def plotPosterior(fnamepfx, trace):
    xSize, ySize = 12, 6
    axes = az.plot_posterior(trace, var_names=['intercept', 'preScale'], figsize=(xSize,ySize))
    for ax in axes.flatten():
        plt.grid(c='grey')
        ax.set_facecolor('white')
    plt.savefig(f'{fnamepfx}-posterior.png')
    plt.savefig(f'{fnamepfx}-posterior.pdf')
    plt.close()

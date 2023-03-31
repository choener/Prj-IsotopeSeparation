
from random import shuffle
from scipy import stats
import aesara
import aesara.tensor as at
import arviz as az
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy
import xarray as xr
import random

import Stats
from Construct import SummaryStats
import Kmer

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 1000
#aesara.config.profile = True

def genKcoords (k):
  assert (k>0)
  if k==1:
    return [ 'A', 'C', 'G', 'T' ]
  else:
    s1 = genKcoords(1)
    sk = genKcoords(k-1)
    return [ s+ss for s in s1 for ss in sk ]

# The holy model:
# - logistic regression on 0, 30, 100 % deuterium; i.e 0; 0.3; 1.0
# - individual data can be pulled up or down by the pre-median calibration
# - the k1med and friends provide scaling on individual data points
# - deuterium ~ logistic (k1med * (x - premedian * PMS))
#
# - possibly set sigma to mad

# TODO consider normalization

def runModel(kmer, df, train = True, posteriorpredictive = True, priorpredictive = True):
  # prepare subsampling
  rels = df['rel'].value_counts()
  samplecount = int(min(rels) / (4**int(kmer)))
  sampledreads = []
  for i in rels.index:
    cands = list(set(df[df['rel']==i].droplevel('k').index))
    random.shuffle(cands)
    sampledreads.extend(cands[0:samplecount])
  log.info(f'subsampled {samplecount} reads for each d2o level')
  #ks = set(df['k'])
  #rels= set(df['rel'])
  #assert len(ks)>0
  #kmerlen = len(list(ks)[0])
  #log.info(f'Model with {len(ks)} kmers of size {kmerlen} and rels: {rels}')

  # The "madZ" values are all positive. We apply a Box-Cox transformation here
  meanmadz = df[df['madZ']>0]['madZ'].mean()
  df['madZ'] = df['madZ'].apply(lambda x: x if x > 0 else meanmadz)
  mads, lmbd = scipy.stats.boxcox(df['madZ'])
  df = df.assign(madZbc = mads)

  # transform the column into the correct matrix form
  medianZ = df['medianZ'].to_xarray()
  madZbc = df['madZbc'].to_xarray()
  rel = df['rel'].to_xarray()[:,0]
  rel = rel.drop_vars('k')

  ## build up the actual model
  #kMedians = None
  #kMads = None
  #if kmer=='k1':
  #  kMedians = k1Medians
  #elif kmer=='k3':
  #  kMedians = k3Medians
  #  # k3mads
  #  tmp = k3Mads.flatten()
  #  tmp[tmp == 0] = np.median(tmp[tmp>0])
  #  # TODO better plots, separating out the classes!
  #  _, ax = plt.subplots(figsize=(12, 6))
  #  az.plot_kde(tmp, label='MAD '+kmer, bw='scott', plot_kwargs={'color':'black'})
  #  kMads, lmbd = scipy.stats.boxcox(tmp)
  #  kMads = (kMads - np.mean(kMads)) / np.std(kMads)
  #  az.plot_kde(kMads, label=f'BoxCox, λ={lmbd :.2f}', bw='scott', plot_kwargs={'color':'blue'})
  #  ax.legend(fontsize=10, frameon=True, framealpha=0.5)
  #  plt.xlim(left=-5, right=10)
  #  plt.savefig(f'{kmer}-kmads.png')
  #  plt.savefig(f'{kmer}-kmads.pdf')
  #  plt.close()
  #  kMads = kMads.reshape(-1,64)
  #  # k3len
  #  tmp = k3Len.flatten()
  #  tmp[tmp == 0] = np.median(tmp[tmp>0])
  #  _, ax = plt.subplots(figsize=(12, 6))
  #  az.plot_kde(tmp)
  #  kLen, lmbd = scipy.stats.boxcox(tmp)
  #  kLen = (kLen - np.median(kLen)) / np.std(kLen)
  #  az.plot_kde(kLen)
  #  plt.xlim(left=-5, right=20)
  #  plt.savefig(f'{kmer}-klen.png')
  #  plt.savefig(f'{kmer}-klen.pdf')
  #  plt.close()
  #  kLen = kLen.reshape(-1,64)
  #elif kmer=='k5':
  #  kMedians = k5Medians
  # prepare coords
  coords = { 'kmer': Kmer.gen(int(kmer))
           }
  #
  # prepare model
  #
  with pm.Model(coords = coords) as model:
    # data we want to be able to swap for posterior predictive
    # access via get_value() / set_value()
    #preMedian = pm.MutableData("preMedian", preMedian.reshape(-1,1))
    medianZ  = pm.MutableData("medianZ", medianZ) #np.array([...])
    madZbc   = pm.MutableData('madZbc', madZbc) #np.array(kMads))

    #pScale    = pm.Beta('preScale', 0.5, 0.5)
    kScale    = pm.Normal('scale', 0, 1, dims='kmer')
    mScale    = pm.Normal('mad', 0, 1, dims='kmer')
    intercept = pm.Normal('intercept', 0, 10)

    rowSum    =  pm.math.dot(medianZ, kScale)
    #rowSum    =  pm.math.dot(medianZ - pScale * preMedian, kScale)
    rowSum    += pm.math.dot(madZbc, mScale)
    predpcnt  = pm.Deterministic('p', pm.math.invlogit(intercept + rowSum))

    log.info(f'medianZ data shape: {medianZ.get_value().shape}')
    log.info(f'madZbc data shape: {madZbc.get_value().shape}')
    log.info(f'scale shape: {kScale.shape}')
    log.info(f'obs shape: {rel.shape}')
    log.info(f'obs: {rel}')

    #obs = pm.Normal("obs", mu=predpcnt, sigma=err, observed=pcnt)
    obs = pm.Bernoulli("obs", p=predpcnt, observed=rel)

  # prior predictive checks needs to be written down still
  if priorpredictive:
    pass
    #with model:
    #  log.info('running prior predictive model')
    #  trace = pm.sample_prior_predictive()
    #  _, ax = plt.subplots()
    #  x = xr.DataArray(np.linspace(-5,5,10), dims=["plot_dim"])
    #  prior = trace.prior
    #  ax.plot(x, x)
    #  plt.savefig(f'{kmer}-prior-predictive.jpeg')
    #  plt.close()

  trace = az.InferenceData()
  if train:
    with model:
      log.info('training model')
      trace = pm.sample(1000, return_inferencedata=True, tune=1000, chains=2, cores=2)
      trace.to_netcdf(f'{kmer}-trace.netcdf')

      # TODO pickle the trace
  else:
    # TODO possibly load model
    trace = trace.from_netcdf(f'{kmer}-trace.netcdf')
    pass
  # create plot(s); later guard via switch
  if False:
    # plot only subset?
    az.plot_trace(trace, compact=True, combined=True, var_names=['~p']) # 'intercept', 'pScale', 'scale'+kmer])
    plt.savefig(f'{kmer}-trace.png')
    plt.savefig(f'{kmer}-trace.pdf')
    plt.close()
    #az.plot_forest(trace, var_names=['~p'])
    #plt.savefig(f'{kmer}-forest.png')
    #plt.savefig(f'{kmer}-forest.pdf')
    #plt.close()
    print(az.summary(trace, var_names=['~p'], round_to=2))

  # plot the posterior, should be quite fast
  # TODO only plots subset of figures, if there are too many subfigures
  log.info(f'plotting posterior distributions')
  #g = medianZ.get_value().shape[1]
  #g = 1 + int(np.sqrt(g+2))
  az.plot_posterior(trace, var_names=['intercept']) # , 'preScale']) # , grid=(g,g))
  plt.savefig(f'{kmer}-posterior.png')
  plt.savefig(f'{kmer}-posterior.pdf')
  plt.close()
  #az.plot_posterior(trace, var_names=['~p', '~intercept', '~preScale'])
  #plt.savefig(f'{kmer}-posterior-all.png')
  #plt.savefig(f'{kmer}-posterior-all.pdf')
  #plt.close()

  if posteriorpredictive:
    with model:
      log.info('posterior predictive run')
      assert trace is not None
      # TODO
      # Normally, we should now go and set new data via
      # pm.set_data({"pred": out-of-sample-data})
      # but we can pickle the trace, then reload with new data
      trace = pm.sample_posterior_predictive(trace, var_names=['p', 'obs'], return_inferencedata=True, extend_inferencedata=True, predictions=True)
      print(trace)
      # important: contains "p"
      mpreds = trace['predictions']
      mppmean = mpreds['p'].mean(axis=(0,1))
      mppmean = mppmean.rename({'p_dim_2' : 'read'})
      mppstd = mpreds['p'].std(axis=(0,1))
      mppstd = mppstd.rename({'p_dim_2' : 'read'})
      print('mppmean', mppmean)
      print('mppmean.coords', mppmean.coords)
      print('rel', rel)
      obs = xr.DataArray(data=rel) #, coords=mppmean.coords)
      print('obs', obs)
      # inplace sorting of the results, keeps means and variances associated
      mppmean = mppmean.sortby(mppmean)
      mppstd  = mppstd.sortby(mppmean)
      obs     = obs.sortby(mppmean)
      _, ax = plt.subplots(figsize=(12, 6))
      # mean with +- stddev
      ax.plot(mppmean, color='blue')
      ax.plot(mppmean + mppstd, color='blue')
      ax.plot(mppmean - mppstd, color='blue')
      ax.plot(obs, 'o')
      # actual
      ax.set_xlabel('Samples (ordered)')
      ax.set_ylabel('Prediction Error')
      ax.set_title('Posterior Predictive Error (±σ)')
      ax.legend(fontsize=10, frameon=True, framealpha=0.5)
      plt.savefig(f'{kmer}-poos.png')
      plt.savefig(f'{kmer}-poos.pdf')
      plt.close()
      ## finally draw for each element, how good the prediction went.
      ## TODO should have multiple lines, depending on 0%, 100%, etc
      ## positive: 0% with their errors
      ## negative: 100% with their negated errors
      #aom = (mppmean-obs)
      #p0 = aom.where(lambda x: x >= 0, drop=True)
      #p0 = p0.sortby(p0)
      #p0good = len(p0.where(lambda x: x < 0.5, drop=True))
      #p1 = abs(aom.where(lambda x: x <  0, drop=True))
      #p1 = p1.sortby(p1)
      #p1good = len(p1.where(lambda x: x < 0.5, drop=True))
      ##aom = aom.sortby(aom)
      ##lastgoodaom = aom.where(lambda x: x < 0.5, drop=True)
      ##print(len(lastgoodaom))
      #_, ax = plt.subplots(figsize=(12, 6))
      #plt.axhline(y=0.5, color='black', linestyle='-')
      #ax.plot(p0, color='orange', label='0% D2O')
      #plt.axvline(x=p0good, color='orange', linestyle='solid')
      #plt.annotate(f'{p0good / len(p0):.2f}', xy=(p0good,0.6), color='orange')
      #ax.plot(p1, color='blue', label='100% D2O')
      #plt.axvline(x=p1good, color='blue', linestyle='dashed')
      #plt.annotate(f'{p1good / len(p1):.2f}', xy=(p1good,0.4), color='blue')
      ## horizontal line at error 0.5
      #ax.set_xlabel('Samples (ordered)')
      #ax.set_ylabel('Prediction Error')
      #ax.set_title('Posterior Predictive Error (per sample)')
      #ax.legend(fontsize=10, frameon=True, framealpha=0.5)
      ## TODO vertical line that is annotated with percentage "good"
      #plt.savefig(f'{kmer}-order-qos.png')
      #plt.savefig(f'{kmer}-order-qos.pdf')
      #plt.close()


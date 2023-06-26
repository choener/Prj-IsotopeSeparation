
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
def buildModel(coords, preMedian, medianZ, madZbc, obs, kmer, totalSz, batchSz = 1000):
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
    obs = pm.Bernoulli("obs", p=predpcnt, observed=obs, total_size = totalSz)
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

def runModel(kmer, df, train = True, posteriorpredictive = True, priorpredictive = True, maxsamples = None, sampler = "jax", batchSz=1000):

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
  print(df)

  # The "madZ" values are all positive. We apply a Box-Cox transformation here
  meanmadz = df[df['madZ']>0]['madZ'].mean()
  # TODO Try using .loc[row_indexer,col_indexer] = value instead
  df['madZ'] = df['madZ'].apply(lambda x: x if x > 0 else meanmadz)
  mads, lmbd = scipy.stats.boxcox(df['madZ'])
  df = df.assign(madZbc = mads)

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

  # Minibatch in case of "advi" only
  if sampler=="advi":
    print(rel)
#    preMedian = pm.Minibatch(preMedian, batch_size=500)
#    medianZ = pm.Minibatch(medianZ, batch_size=500)
#    madZbc = pm.Minibatch(madZbc, batch_size=500)
    z = df['rel'].to_xarray()[:,0]
    z = z.drop_vars('k')
    print(">>>")
    print(z)
    print(type(z))
    print("<<<")
    zz = pm.Minibatch(z, batch_size=500)
    print("^^^")
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
    log.info('training model')
    model = None
    mf = None # mean field
    trace = None
    match sampler:
      case "adagrad":
        mbPreMedian, mbMedianZ, mbMadZbc, mbRel = pm.Minibatch(nmPreMedian, nmMedianZ, nmMadZbc, nmRel, batch_size=batchSz)
        model = buildModel(coords, mbPreMedian, mbMedianZ, mbMadZbc, mbRel, kmer, rel.shape)
        # run the model with mini batches
        with model:
          mf : pm.MeanField = pm.fit(obj_optimizer=pm.adagrad_window(learning_rate=1e-2))
          trace = mf.sample(draws=1000)
      case "advi":
        model = buildModel(coords, mbPreMedian, mbMedianZ, mbMadZbc, mbRel, kmer)
        with model:
          mf = pm.fit(method="advi")
          trace = mf.sample(draws=1000)
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
    trace.to_netcdf(f'{kmer}-trace.netcdf')
    # TODO pickle the trace
  else:
    # TODO possibly load model
    trace = trace.from_netcdf(f'{kmer}-trace.netcdf')
    pass
  # create plot(s); later guard via switch
  if True:
    # plot only subset?
    az.plot_trace(trace, compact=True, combined=True, var_names=['~p']) # 'intercept', 'pScale', 'scale'+kmer])
    plt.savefig(f'{kmer}-trace.png')
    plt.savefig(f'{kmer}-trace.pdf')
    plt.close()
    az.plot_forest(trace, var_names=['~p'])
    plt.savefig(f'{kmer}-forest.png')
    plt.savefig(f'{kmer}-forest.pdf')
    plt.close()
    scaleMeans = trace.posterior["scale"].mean(("chain", "draw"))
    sortedTrace = trace.posterior["scale"].sortby(scaleMeans)
    print(az.summary(trace, var_names=['~p'], round_to=2))

  # plot the posterior, should be quite fast
  # TODO only plots subset of figures, if there are too many subfigures
  log.info(f'plotting posterior distributions')
  #g = medianZ.get_value().shape[1]
  #g = 1 + int(np.sqrt(g+2))
  az.plot_posterior(trace, var_names=['intercept', 'preScale']) # , grid=(g,g))
  plt.savefig(f'{kmer}-posterior.png')
  plt.savefig(f'{kmer}-posterior.pdf')
  plt.close()
  #az.plot_posterior(trace, var_names=['~p', '~intercept', '~preScale'])
  #plt.savefig(f'{kmer}-posterior-all.png')
  #plt.savefig(f'{kmer}-posterior-all.pdf')
  #plt.close()

  if posteriorpredictive:
    # rebuild the model with full data!
    #model = buildModel(coords, preMedian, medianZ, madZbc, rel, kmer, rel.shape)
    model = buildModel(coords, nmPreMedian, nmMedianZ, nmMadZbc, nmRel, kmer, nmRel.shape)
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
      _, ax = plt.subplots(figsize=(12, 6))
      # mean with +- stddev
      ax.plot(mppmean, color='blue')
      ax.plot(mppmean + mppstd, color='blue')
      ax.plot(mppmean - mppstd, color='blue')
      ax.plot(obs, '.')
      # actual
      ax.set_xlabel('Samples (sorted by p)')
      ax.set_ylabel('p (Predicted D2O)')
      ax.set_title('Posterior Predictive Error (±σ)')
      ax.legend(fontsize=10, frameon=True, framealpha=0.5)
      plt.savefig(f'{kmer}-poos.png')
      plt.savefig(f'{kmer}-poos.pdf')
      plt.close()
      # finally draw for each element, how good the prediction went.
      # TODO should have multiple lines, depending on 0%, 100%, etc
      # positive: 0% with their errors
      # negative: 100% with their negated errors
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
      #aom = aom.sortby(aom)
      #lastgoodaom = aom.where(lambda x: x < 0.5, drop=True)
      #print(len(lastgoodaom))
      _, ax = plt.subplots(figsize=(12, 6))
      plt.axhline(y=0.5, color='black', linestyle='-')
      ax.plot(p0, color='orange', label='0% D2O')
      plt.axvline(x=p0good, color='orange', linestyle='solid')
      plt.annotate(f'{p0good / len(p0):.2f}', xy=(p0good,0.6), color='orange')
      ax.plot(p1, color='blue', label='100% D2O')
      plt.axvline(x=p1good, color='blue', linestyle='dashed')
      plt.annotate(f'{p1good / len(p1):.2f}', xy=(p1good,0.4), color='blue')
      # horizontal line at error 0.5
      ax.set_xlabel('Samples (ordered)')
      ax.set_ylabel('Prediction Error')
      ax.set_title('Posterior Predictive Error (per sample)')
      ax.legend(fontsize=10, frameon=True, framealpha=0.5)
      # TODO vertical line that is annotated with percentage "good"
      plt.savefig(f'{kmer}-order-qos.png')
      plt.savefig(f'{kmer}-order-qos.pdf')
      plt.close()


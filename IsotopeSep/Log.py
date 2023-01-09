
import aesara.tensor as at
import aesara
import arviz as az
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import xarray as xr
from random import shuffle

from Construct import SummaryStats

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
az.rcParams["plot.max_subplots"] = 200
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

def runModel(stats : SummaryStats, kmer, train = True, posteriorpredictive = True, priorpredictive = True):
  assert (kmer=='k1' or kmer=='k3' or kmer=='k5')
  # explicit selection of which rows to keep around, removing all with errors and such things
  keep = []
  for i,k in enumerate(stats.label):
    l=float(k)
    if l>=0 and l<=1 and not np.isnan(stats.preMedian[i]):
      keep.append(True)
    else:
      keep.append(False)
  keep = np.array(keep)
  keepcnt = {}
  for k in np.array(stats.label)[keep]:
    keepcnt[k] = keepcnt.get(k, 0) + 1
  log.info(f'counts in stats: {keepcnt}')
  minkeep = min(keepcnt.values())
  log.info(f'keep at most: {minkeep}')
  #
  # TODO select the min(keepcnt), then split keep into classes, for each class draw min(keepcnt)
  # elements
  #
  selcnt = {}
  for k in keepcnt.keys():
    selcnt[k] = 0
  shuffled = list(enumerate(stats.label))
  shuffle(shuffled)
  for i,k in shuffled:
    l = float(k)
    if l > 0 and l < 1:
      keep[i] = False
    if keep[i] and selcnt[k] < minkeep:
      selcnt[k] = selcnt[k] + 1
    elif k >= 0:
      keep[i] = False
  log.info(f'counts after minimal selection: {selcnt}, checking with: {sum(keep)}')

  #
  # prepare data based on what to keep
  #
  # percent deuterium
  pcnt = np.array([float(x) / 1.0 for x in stats.label])[keep]
  # N (vector)
  preMedian = np.array(stats.preMedian)[keep] # ,ndmin=2).T[keep]
  # Nx4^1
  k1Medians = np.array(stats.k1Medians)[keep]
  # Nx4^3
  k3Medians = np.array(stats.k3Medians)[keep]
  # Nx4^5
  k5Medians = np.array(stats.k5Medians)[keep]
  # rescale
  k1mean = np.mean(k1Medians)
  k1var  = np.std(k1Medians)
  k3mean = np.mean(k3Medians)
  k3var  = np.std(k3Medians)
  k5mean = np.mean(k5Medians)
  k5var  = np.std(k5Medians)
  # global pulldown
  preMedian = preMedian - k5mean
  k1Medians = k1Medians - k5mean # - preMedian.reshape(-1,1) # (k1Medians - k1mean) / k1var
  k3Medians = k3Medians - k5mean #(k3Medians - k3mean) / k3var
  k5Medians = k5Medians - k5mean #(k5Medians - k5mean) / k5var
  # for generic model
  kMedians = None
  kLen = None
  if kmer=='k1':
    kMedians = k1Medians
    kLen = np.array(stats.k1LenMean)[keep]
  elif kmer=='k3':
    kMedians = k3Medians
    kLen = np.array(stats.k3LenMean)[keep]
  elif kmer=='k5':
    kMedians = k5Medians
    kLen = np.array(stats.k3LenMean)[keep]
  # prepare coords
  coords = { 'k1': genKcoords(1)
           , 'k3': genKcoords(3)
           , 'k5': genKcoords(5)
           , 'kmer': genKcoords(int(kmer[1]))
           }
  #
  # prepare model
  #
  with pm.Model(coords = coords) as model:
    # data we want to be able to swap for posterior predictive
    # access via get_value() / set_value()
    preMedian = pm.MutableData("preMedian", preMedian.reshape(-1,1))
    kMedians  = pm.MutableData("kMedians", np.array(kMedians))

    pScale    = pm.Beta('preScale', 0.5, 0.5)
    kScale    = pm.Normal('scale'+kmer, 0, 10, dims='kmer')
    intercept = pm.Normal('intercept', 0, 10)
    #err       = pm.HalfNormal("err",sigma=1)

    rowSum    = pm.math.dot(kMedians - pScale * preMedian, kScale)
    predpcnt  = pm.Deterministic('p', pm.math.invlogit(intercept + rowSum))

    log.info(f'{kMedians.get_value().shape}')
    log.info(f'{kScale.shape}')

    #obs = pm.Normal("obs", mu=predpcnt, sigma=err, observed=pcnt)
    obs = pm.Bernoulli("obs", p=predpcnt, observed=pcnt)

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
    print(az.summary(trace, var_names=['intercept', 'scale'+kmer], round_to=2))

  # plot the posterior, should be quite fast
  # TODO only plots subset of figures, if there are too many subfigures
  log.info(f'plotting posterior distributions')
  #g = kMedians.get_value().shape[1]
  #g = 1 + int(np.sqrt(g+2))
  az.plot_posterior(trace, var_names=['intercept', 'preScale']) # , grid=(g,g))
  plt.savefig(f'{kmer}-posterior.png')
  plt.savefig(f'{kmer}-posterior.pdf')
  plt.close()
  az.plot_posterior(trace, var_names=['scale'+kmer]) # , grid=(g,g))
  plt.savefig(f'{kmer}-posterior-all.png')
  plt.savefig(f'{kmer}-posterior-all.pdf')
  plt.close()

  if posteriorpredictive:
    with model:
      log.info('posterior predictive run')
      assert trace is not None
      # TODO
      # Normally, we should now go and set new data via
      # pm.set_data({"pred": out-of-sample-data})
      # but we can pickle the trace, then reload with new data
      trace = pm.sample_posterior_predictive(trace, var_names=['p', 'obs'], return_inferencedata=True, extend_inferencedata=True, predictions=True)
      # important: contains "p"
      mpreds = trace['predictions']
      mppmean = mpreds['p'].mean(axis=(0,1))
      mppstd = mpreds['p'].std(axis=(0,1))
      obs = xr.DataArray(data=pcnt, coords=mppmean.coords)
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
      # finally draw for each element, how good the prediction went.
      # TODO should have multiple lines, depending on 0%, 100%, etc
      # positive: 0% with their errors
      # negative: 100% with their negated errors
      aom = (mppmean-obs)
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



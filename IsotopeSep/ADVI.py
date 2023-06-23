
# ADVI implementation of the holy model. For larger (>3) kmer lengths, the full Bayesian
# implementation does not converge in time.

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import Kmer

"""
Construct the model with this function. This allows us to hand in either full data, or minibatches
as necessary.
"""
def constructModel(coords):
  with pm.Model(coords=coords) as model:
    return model

"""
The ADVI variant of the model. We use Minibatch to keep the compute times reasonable. This requires
pymc 5.5, due to bugs present in Minibatch earlier than 5.5.

coords are the coordinate names for kmers.
"""

def trainADVI(preMedian, medianZ, madZbc, obs, kmer, batchSz = 1000):
  coords = { 'kmer': Kmer.gen(int(kmer))
           }
  # construct model
  with pm.Model(coords = coords) as advi:
    # construct minibatches
    mPreMedian, mMedianZ, mMadZbc = pm.Minibatch(preMedian, medianZ, madZbc, batch_size=batchSz)
    # RVs
    pScale    = pm.Beta('preScale', 0.5, 0.5)
    kScale    = pm.Normal('scale', 0, 1, dims='kmer')
    mScale    = pm.Normal('mad', 0, 1, dims='kmer')
    intercept = pm.Normal('intercept', 0, 10)
    rowSum    =  pm.math.dot(medianZ - pScale * preMedian, kScale)
    rowSum    += pm.math.dot(madZbc, mScale)
    predpcnt  = pm.Deterministic('p', pm.math.invlogit(intercept + rowSum))
    obs = pm.Bernoulli("obs", p=predpcnt, observed=obs, total_size = obs.shape)

  # fit model, create trace
  trace = None
  with advi:
    mf : pm.MeanField = pm.fit(obj_optimizer=pm.adagrad_window(learning_rate=1e-2))
    #mf : pm.MeanField = pm.fit()
    plt.xscale('log')
    plt.plot(mf.hist)
    plt.savefig(f'advi-{kmer}-hist.png')
    trace = mf.sample(draws=1000)
    print(pm.summary(trace))
    pm.plot_trace(trace)
    plt.savefig(f'advi-{kmer}-trace.png')
    pm.plot_posterior(trace)
    plt.savefig(f'advi-{kmer}-posterior.png')

  # TODO this needs to re-run the model, but with full data, not minibatch data!
  #with model(D,O):
  #  ppc = pm.sample_posterior_predictive(trace, var_names=['logit'], return_inferencedata=True, extend_inferencedata=True, predictions=True)
  #  ppcMeanLogit = np.sort(abs(O - ppc['predictions']['logit'].mean(axis=(0,1))))
  #  _, ax = plt.subplots(figsize=(12, 6))
  #  ax.plot(ppcMeanLogit)
  #  plt.savefig('advi-ppcMeanLogit.png')

"""
Test the model with new data. Makes use of a previously fitted model
"""

def testADVI():
  pass

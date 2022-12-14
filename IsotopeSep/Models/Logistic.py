
# Logistic model for isotope separation

from pymc import Model, Normal, HalfCauchy, sample, Dirichlet, HalfNormal, MutableData
import aesara as ae
import aesara.tensor as at
import numpy as np
import pymc as mc

#from Models.Common import *

# This model
#
# NOTE a variant with 3-mers is possible but leads to divergence of the sampler somewhat often
# TODO normalize everything; but over all data.

def modelLogistic(pdAll, k):
  pd, test = pdSplit(pdAll)
  ys = np.array(pd['means'])
  ls = np.array(pd['labels'])
  xs1 = np.asmatrix(np.vstack(pd['n1rel'].values))
  xs2 = np.asmatrix(np.vstack(pd['n2rel'].values))
  xs3 = np.asmatrix(np.vstack(pd['n3rel'].values))
  _, cols1 = xs1.shape
  _, cols2 = xs2.shape
  _, cols3 = xs3.shape
  Tys = np.array(test['means'])
  Tls = np.array(test['labels'])
  Txs1 = np.asmatrix(np.vstack(test['n1rel'].values))
  Txs2 = np.asmatrix(np.vstack(test['n2rel'].values))
  Txs3 = np.asmatrix(np.vstack(test['n3rel'].values))
  xs = None
  Txs = None
  if k==1:
    xs = xs1
    Txs = Txs1
  if k==2:
    xs = xs2
    Txs = Txs2
  if k==3:
    xs = xs3
    Txs = Txs3
  assert xs is not None
  rows, cols = xs.shape
  with Model() as model:
    YS = MutableData('YS', ys)
    XS = MutableData('XS', xs)
    LS = MutableData('LS', ls)
    # should be at zero, since we normalized
    mu = Normal('μ',0, sigma=1)
    beta = Normal('β', mu = np.zeros(cols), sigma=10, shape=(cols))
    ll = YS + mu + at.dot(XS, beta)
    p = mc.Deterministic('p', mc.invlogit(ll))
    likelihood = mc.Bernoulli('obs', p=p, observed = LS)
    trace = memoize(f'logistic-{k}.model', model)
  print('sampling finished')
  varNames = ['μ', 'β']
  #az.plot_trace(trace,figsize=(20,20))
  #plt.savefig(f'logistic-{k}-trace.pdf')
  az.plot_posterior(trace, var_names = varNames,figsize=(20,20))
  plt.savefig(f'logistic-{k}-posterior.pdf', bbox_inches='tight')
  s = az.summary(trace, var_names = varNames)
  print(s)
  print(trace['posterior'])
  #
  print('posterior predictive')
  with model:
    zeros=test[test['labels']==0]
    print(len(zeros))
    mc.set_data({'XS': Txs, 'YS': Tys})
    print(Txs.shape, Tls.shape)
    # every 30th only
    test = mc.sample_posterior_predictive(trace.sel(draw=slice(None,None,30)), var_names=['p'])
    ppc = test['posterior_predictive']
    # mean prediction over all posterior predictive samples (from all chains)
    predls = test['posterior_predictive']['p'].mean(axis=0).mean(axis=0)
    predls0 = predls[:len(zeros)]
    predls1 = predls[len(zeros):]
    # tiny effect, but the effect is there, even in the test set!
    print(predls0.mean(),predls1.mean())
    plt.figure(figsize=(20,4))
    plt.plot(Tls)
    #plt.plot(ppc['p'].mean(axis=0).mean(axis=0))
    plt.plot(np.concatenate([np.sort(predls0),np.sort(predls1)]))
    plt.savefig(f'logistic-{k}-postpred.pdf', bbox_inches='tight')


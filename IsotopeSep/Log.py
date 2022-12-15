
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from Construct import SummaryStats

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# The holy model:
# - logistic regression on 0, 30, 100 % deuterium; i.e 0; 0.3; 1.0
# - individual data can be pulled up or down by the pre-median calibration
# - the k1med and friends provide scaling on individual data points
# - deuterium ~ logistic (k1med * (x - premedian * PMS))
#
# - possibly set sigma to mad

# TODO consider normalization

def runModel(stats : SummaryStats):
  coords = { 'k1': ['A','C','G','T'] }
  k1 = int(4**1)
  with pm.Model(coords = coords) as model:
    keep = np.array(stats.label)
    keep = keep>=0
    preMedian = pm.Data("preMedian", np.array(stats.preMedian,ndmin=2).T[keep])
    k1Medians = pm.Data("k1Medians", np.array(stats.k1Medians)[keep])
    pcnt = pm.Data("label", np.array(stats.label)[keep])

    # k1Medians, but pulled down, pull-down scale depends on "pms"
    pms = pm.Normal("PMS",mu=0, sigma=1, shape=1)
    k1Z = pm.Deterministic("k1Z", k1Medians - pms * preMedian)
    k1I = pm.Normal("k1I", mu=0, sigma=10, shape=[k1])
    k1S = pm.Normal("k1S", mu=0, sigma=1, shape=[k1])

    # the actual logistic model for each row of data
    rowSum = (k1S * (k1Z-k1I)).sum(axis=1)
    invlog = pm.Deterministic("ll",pm.invlogit(rowSum))
    print(model.basic_RVs)




#    preMedian = pm.Data("preMedian", np.ndarray(stats.preMedian[keep]))
#    k1Medians = pm.Data("k1Medians", np.matrix(stats.k1Medians)[keep])
#    pcnt = pm.Data("pcnt", np.array(stats.label)[keep])
#    # pre-median scale: this times the preMedian is the scaler, if zero then we don't care about
#    # the pre-median
#    preScale = pm.Normal("PMS", mu=0, sigma=1, shape=1)
#    # this generates new data
#    x1Medians = k1Medians - preMedian
#    print(x1Medians)
#    k1med = pm.Normal("k1med", mu=0, sigma=1, shape=k1)
#    mu = pm.Normal("mu", mu=0, sigma=1)
#    obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))
#    print(model.basic_RVs)
#    #ll = pm.Deterministic("ll", pm.invlogit(xs), observed = deuteriumProb)


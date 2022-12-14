
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# The holy model:
# - logistic regression on 0, 30, 100 % deuterium; i.e 0; 0.3; 1.0
# - individual data can be pulled up or down by the pre-median calibration
# - deuterium ~ logistic (x - premedian * PMS)

def runModel():
  coords = { 'k1': ['A','C','G','T'] }
  k1 = int(4**1)
  k3 = int(4**3)
  k5 = int(4**5)
  with pm.Model(coords = coords) as model:
    # pre-median scale: this times the preMedian is the scaler, if zero then we don't care about
    # the pre-median
    preScale = pm.Normal("PMS", mu=0, sigma=1, shape=1)
    k1med = pm.Normal("k1med", mu=0, sigma=1, shape=k1)
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=rng.standard_normal(100))
    print(model.basic_RVs)
    pass

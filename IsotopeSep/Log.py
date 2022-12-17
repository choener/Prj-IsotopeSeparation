
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from Construct import SummaryStats, kmer2int

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

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

def runModel(stats : SummaryStats, kmer):
  assert (kmer=='k1' or kmer=='k3' or kmer=='k5')
  # explicit selection of which rows to keep around, removing all with errors and such things
  keep = []
  for i,k in enumerate(stats.label):
    l=float(k)
    if l>=0 and l<=100 and not np.isnan(stats.preMedian[i]):
      keep.append(True)
    else:
      keep.append(False)
  keep = np.array(keep)
  #
  # prepare data based on what to keep
  #
  # percent deuterium
  pcnt = np.array([float(x) / 100.0 for x in stats.label])[keep]
  preMedian = np.array(stats.preMedian)[keep] # ,ndmin=2).T[keep]
  k1Medians = np.array(stats.k1Medians)[keep]
  k3Medians = np.array(stats.k3Medians)[keep]
  k5Medians = np.array(stats.k5Medians)[keep]
  # rescale everything by the k5 values, should be roughly correct
  k5mean = np.mean(k5Medians)
  k5var  = np.std(k5Medians)
  k1Medians = (k1Medians - k5mean) / k5var
  k3Medians = (k3Medians - k5mean) / k5var
  k5Medians = (k5Medians - k5mean) / k5var
  # for generic model
  kMedians = None
  if kmer=='k1':
    kMedians = k1Medians
  elif kmer=='k3':
    kMedians = k3Medians
  elif kmer=='k5':
    kMedians = k5Medians
  # normalize
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
    preMedian = pm.MutableData("preMedian", preMedian)
    #k1Medians = pm.MutableData("k1Medians", np.array(k1Medians))
    #k3Medians = pm.MutableData("k3Medians", np.array(k3Medians))
    #k5Medians = pm.MutableData("k5Medians", np.array(k5Medians))

    kMedians = pm.MutableData("kMedians", np.array(kMedians))
    kScale = pm.Normal(kmer + 'Scale', 0, 1, dims='kmer')

    print(kScale.shape)
    print(kMedians.get_value().shape)

    #rowSum = np.zeros(shape=len(pcnt))
    #for i,n in enumerate(coords[kmer]):
    #  rowSum += kScale[kmer2int(n)] * kMedians[:,kmer2int(n)]

    rowSum = pm.math.dot(kMedians, kScale)

    #k1 = pm.Normal('k1Scale', 0, 1, dims='k1')
    #k3 = pm.Normal('k3Scale', 0, 1, dims='k3')
    #k5 = pm.Normal('k5Scale', 0, 1, dims='k5')

    #rowSum1 = np.zeros(shape=(len(pcnt)))
    #for i,n in enumerate(coords['k1']):
    #  rowSum1 += k1[kmer2int(n)] * k1Medians[:,kmer2int(n)]
    #rowSum3 = np.zeros(shape=(len(pcnt)))
    #for i,n in enumerate(coords['k3']):
    #  rowSum3 += k3[kmer2int(n)] * k3Medians[:,kmer2int(n)]
    #rowSum5 = np.zeros(shape=(len(pcnt)))
    #for i,n in enumerate(coords['k5']):
    #  rowSum5 += k5[kmer2int(n)] * k5Medians[:,kmer2int(n)]

    intercept = pm.Normal('intercept', 0, 5)
    predpcnt = pm.math.invlogit(intercept + rowSum)

    err = pm.HalfNormal("err",sigma=1)

    pm.Normal("obs", mu=predpcnt, sigma=err, observed=pcnt)

    trace = pm.sample(1000, tune=1000, cores=1)

  az.plot_trace(trace,figsize=(20,20))
  plt.savefig(f'{kmer}-log-trace.jpeg')


#    # k1Medians, but pulled down, pull-down scale depends on "pms"
#    pms = pm.Normal("PMS",mu=0, sigma=1, shape=1)
#    k1Z = pm.Deterministic("k1Z", k1Medians - pms * preMedian)
#    k1I = pm.Normal("k1I", mu=0, sigma=10, shape=[k1])
#    k1S = pm.Normal("k1S", mu=0, sigma=1, shape=[k1])
#
#    predpcnt = pm.math.invlogit(undefined)
#
#    #obs ~ Distribution(...)
#    pm.Bernoulli("obs", p = predpcnt, observed = pcnt)
#    # the actual logistic model for each row of data
#    #rowSum = (k1S * (k1Z-k1I)).sum(axis=1)
#    #invlog = pm.Deterministic("ll",pm.invlogit(rowSum))
#    #print(model.basic_RVs)




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



# Old code


# Statistics for movetabled nucleotide positions in the raw signal

def signalStatistics (splitsignal):
  stats = []
  for ns in splitsignal:
    stats.append([np.mean(ns),statistics.variance(ns),np.median(ns)])
  return np.transpose(np.array(stats))

# Histogram of the relative nucleotide compositions, divided by label. We want to make sure that we
# don't accidentally condition on the nucleotide composition, instead of on the deuterium
# composition.

def nucleotideHistogram (pd):
  # TODO plot n1rel histogram, but separate for each label !
  pass

# TODO run a simple mono- and dinucleotide model for the mean of the signal. Do this for
# raw data
# z-score data
# EMA data
# beta * [A-count,C-count,G-count,T-count,1] ~ observed mean
# now, we need to add the change due to deuterium, which might be a shift in beta?

def modelDirichletNucs (pd):
  with Model():
    ys = np.array(pd['means'])
    ls = np.array(pd['labels'])
    #
    xs1 = np.asmatrix(np.vstack(pd['n1rel'].values))
    _, cols1 = xs1.shape
    print(cols1)
    baseDisp1  = Dirichlet('bs dsp 1', np.ones(cols1))
    baseScale1 = Normal('bs scl 1', 0, sigma=3)
    deutDisp1  = Dirichlet('dt dsp 1', np.ones(cols1))
    deutScale1 = Normal('dt scl 1', 0, sigma=3)
    mu = np.mean(pd['means'])
    mu += baseScale1 * at.dot(xs1,baseDisp1)
    mu += deutScale1 * at.dot(xs1,deutDisp1) * ls
    #
    #xs2 = Data('xs2', value = xsMat2)
    #rows2, cols2 = xsMat2.shape
    #baseDisp2  = Dirichlet('bs dsp 2', np.ones(cols2))
    #baseScale2 = Normal('bs scl 2', 0, sd=3)
    #deutDisp2  = Dirichlet('dt dsp 2', np.ones(cols2))
    #deutScale2 = Normal('dt scl 2', 0, sd=3)
    #mu += baseScale2 * tt.dot(xs2,baseDisp2)
    #mu += deutScale2 * tt.dot(xs2,deutDisp2) * ls
    #
    epsilon = HalfCauchy('ε', 5)
    likelihood = Normal('ys', mu = mu, sigma = epsilon, observed = ys)
    trace = sample(1000, return_inferencedata = True, init="adapt_diag")
    #traceDF = trace_to_dataframe(trace)
    #print(traceDF.describe())
    #print(traceDF['bs scl 1'] < traceDF['dt scl 1'])
    #scatter_matrix(traceDF, figsize=(8,8))
    #traceplot(trace)
    az.plot_posterior(trace)
    plt.savefig('posterior.pdf', bbox_inches='tight')
    az.plot_trace(trace)
    plt.savefig('trace.pdf', bbox_inches='tight')
    # TODO extract the full trace so that I can run a prob that deutscale != 0
    # TODO extract statistics on nucleotide distribution, compare between classes to make sure we
    # don't accidentally train just on that
    #prob_diff = np.mean(trace[:]['bs scl 1'] < trace[:]['dt scl 1'])
    #print('P(mean_base < mean_deut) = %.2f%%' % (prob_diff * 100))

# Normalizes all pandas data, return the new pd frame and the mean,sigma.
# NOTE we normalize on canonical data to make the difference more clear

def normalize(pdd):
  pd = pandas.concat([pdd[pdd['labels']==0], pdd[pdd['labels']==1]])
  mean = pd['means'].mean() # pd[pd['labels']==0]['means'].mean()
  stddev = pd['means'].std()
  pd['means'] = (pd['means'] - mean) / stddev
  print(len(pd))
  print(len(pd[pd['labels']==0]))
  return pd, mean, stddev

# NOTE a variant with 3-mers is possible but leads to divergence of the sampler somewhat often

def modelMixtureNucs(pdAll, k):
  pd, test = pdSplit(pdAll)
  ys = np.array(pd['means'])
  ls = np.array(pd['labels'])
  xs1 = np.asmatrix(np.vstack(pd['n1rel'].values))
  xs2 = np.asmatrix(np.vstack(pd['n2rel'].values))
  xs3 = np.asmatrix(np.vstack(pd['n3rel'].values))
  _, cols1 = xs1.shape
  _, cols2 = xs2.shape
  _, cols3 = xs3.shape
  xs = None
  if k==1:
    xs = xs1
  if k==2:
    xs = xs2
  if k==3:
    xs = xs3
  rows, cols = xs.shape
  with Model() as model:
    beta = Normal('β', mu = np.zeros(cols), sigma = 1, shape=(cols))
    #deut = Normal('δ', mu = np.zeros(cols), sigma = 1, shape=(cols))
    deut = Normal('δ', mu = 0, sigma = 1) # , shape=(1)) # make this scalar (again)
    ll  = at.dot(4*xs-1, beta)
    ll += deut * ls
    error = HalfCauchy('ε', beta = 1)
    likelihood = Normal('ys', mu = ll, sigma = error, observed = ys,shape=(rows))
    trace = memoize(f'mixture-{k}.model', model)
  print('sampling finished')
  #
  varNames = ['ε', 'β', 'δ']
  az.plot_trace(trace,figsize=(20,20))
  plt.savefig(f'mixture-{k}-trace.pdf', bbox_inches='tight')
  az.plot_posterior(trace, var_names = varNames,figsize=(30,20), textsize=26)
  plt.savefig(f'mixture-{k}-posterior.pdf', bbox_inches='tight')
  #
  s = az.summary(trace, var_names = varNames)
  print(s)
  #
  # plot on y=0 all samples with label 0, y=1, label 1; x-axis should have adjusted mu
  #
  return {} # return { 'mu': mu, 'beta1': beta1, 'deut': deut, 'err': error }

# TODO Needs a mapping 'label' -> 'string label', since label will end up being a 0/1 vector for
# classification

def genSummaryStats(labelInformation, xs):
  return pandas.DataFrame ({
    'label': [],
    # prefix information
    'preMedian': [np.median(x) for x in xs['preSignals']],
    # suffix information
    'sufMedian': [np.median(x) for x in xs['mainSignals']],
    'sufMean': [np.mean(x) for x in xs['mainSignals']],
    'sufVar': [np.var(x) for x in xs['mainSignals']],
  })

# Calculates the relative nucleotide composition of a read.
# TODO move the plotting functionality somewhere else!

def relNucComp(pd):
  ks1 = set()
  ks2 = set()
  ks3 = set()
  for n in pd['nstats']:
    for k in n.k1.keys():
      ks1.add(k)
    for k in n.k2.keys():
      ks2.add(k)
    for k in n.k3.keys():
      ks3.add(k)
  ks1=list(ks1)
  ks1.sort()
  ks2=list(ks2)
  ks2.sort()
  ks3=list(ks3)
  ks3.sort()
  print(ks1)
  print(ks2)
  print(ks3)
  xs1 = []
  xs2 = []
  xs3 = []
  for n in pd['nstats']:
    s1 = sum(n.k1.values())
    s2 = sum(n.k2.values())
    s3 = sum(n.k3.values())
    arr1 = np.array([n.k1[k] / s1 for k in ks1])
    arr2 = np.array([n.k2[k] / s2 for k in ks2])
    arr3 = np.array([n.k3[k] / s3 for k in ks3])
    xs1.append(arr1)
    xs2.append(arr2)
    xs3.append(arr3)
  pd['n1rel'] = xs1
  pd['n2rel'] = xs2
  pd['n3rel'] = xs3
  xs1bs = np.vstack(pd['n1rel'][pd['labels']==0])
  xs1dt = np.vstack(pd['n1rel'][pd['labels']==1])
  _, axes = plt.subplots(2,4, figsize=(40,10))
  for c,lbl in enumerate(ks1):
    ax = axes[0,c]
    ax.set_xlim(0.1,0.4)
    az.plot_posterior({lbl + ' (H2O)': xs1bs[:,c]},ax=ax, textsize=26)
    ax = axes[1,c]
    ax.set_xlim(0.1,0.4)
    az.plot_posterior({lbl + ' (D2O)': xs1dt[:,c]},ax=ax, textsize=26)
  plt.savefig('nucleotide-distributions.pdf', bbox_inches='tight')
  return


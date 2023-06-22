
# ADVI implementation of the holy model. For larger (>3) kmer lengths, the full Bayesian
# implementation does not converge in time.

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

"""
Prepares the data matrix. Due to bugs within the Minibatch implementation -- fixed upstream but not
here yet -- we need to construct a single data matrix which is then deconstructed within the model.
The matrices are stacked as such:
  - relative deuterium content [1xn]
  - preMedian [1xn]
  - medianZ [kxn]
  - madZbc [kxn]
where k = 4 ** kmer-length
      n = number of reads
"""

def constructData():
  pass

coords = None
data = None
bsz = 1000

"""
The ADVI variant of the model
"""

with pm.Model(coords = coords) as advi:
  # each minibatch
  mData = pm.Minibatch(data, batch_size=bsz)
  # the submatrices
  pass


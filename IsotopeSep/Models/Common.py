
# Commond functionality for stats models

from os.path import exists
from pymc import sample
from sklearn.model_selection import train_test_split
import pandas as pandas
import pickle

#

def pdSplit(pd):
  #splitPoint = int(0.8*(len(pd)))
  s = 0.2
  lbl0 = pd[pd['labels']==0]
  lbl1 = pd[pd['labels']==1]
  train0, test0 = train_test_split(lbl0, test_size=s)
  train1, test1 = train_test_split(lbl1, test_size=s)
  return pandas.concat([train0,train1]), pandas.concat([test0,test1])
#  n = min(len(lbl0),len(lbl1))
#  pd0 = lbl0.sample(n=n)
#  pd1 = lbl1.sample(n=n)
#  pd = pandas.concat([pd0,pd1])
#  print(len(pd), len(lbl0), len(lbl1), len(test))
#  return pd[:splitPoint], pd[splitPoint:]

#

def memoize(fname, model):
  if exists(fname):
    with model:
      with open(fname,'rb') as buff:
        trace = pickle.load(buff)
  else:
    with model:
      trace = sample(tune = 1000, draws = 3000, return_inferencedata = True)
      with open(fname, 'wb') as buff:
        pickle.dump(trace,buff)
  return trace


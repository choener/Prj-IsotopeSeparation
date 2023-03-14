
import numpy as np

# Return median and median absolute deviation

def medianMad(xs):
  median = np.median(xs)
  mad = np.median(np.abs(xs-median))
  return median, mad


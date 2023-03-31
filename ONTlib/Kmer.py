
def gen(k):
  s = ['A','C','G','T']
  if k==1:
    return s
  else:
    ts = gen(k-1)
    return [ h+t for h in s for t in ts]


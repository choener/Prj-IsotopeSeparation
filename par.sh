#!/usr/bin/env bash

go() {
  P=`basename $1 fast5`pickle
  echo $2
  echo $1
  echo $P
  ./IsotopeSep/isosep.py \
    --barcode 0 taubert-d2o/barcode14.ids \
    --barcode 30 taubert-d2o/barcode15.ids \
    --barcode 100 taubert-d2o/barcode16.ids \
    --reads $1 \
    --outputs fullrun \
    --disableBayes \
    --picklePath $P \
    --limitreads 10
}

export -f go
parallel --jobs 6 go {} {#} ::: taubert-d2o/reads/*fast5


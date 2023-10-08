#!/usr/bin/env bash

# train models [unless they are trained already]
# $1 is the target directory
# $2 the kmer length [1,3,5]
# $3 the summary dirs to work on
# $4, $5 the zero and one targets [0.0 0.3 1.0]

function train {
  mkdir -p $1
  if [ ! -f ./$1/5-adagrad-trace.netcdf ]
  then
    ./IsotopeSep/isosep.py \
      --barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids \
      --summarydirs $3 \
      --posteriorpredictive \
      --zero $4 --one $5 \
      --outputdir $1 \
      --kmer $2 \
      --sampler adagrad \
      --train
  fi
  # always run a posterior predictive afterwards
  ./IsotopeSep/isosep.py \
    --barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids \
    --summarydirs $3 \
    --posteriorpredictive \
    --zero $4 --one $5 \
    --outputdir $1 \
    --kmer $2 \
    --sampler adagrad
}

# test models
# $1 is the target directory
# $2 is the source for learning
# $3 the kmer length [1,3,5]
# $4 the summary dirs to work on
# $5, $6 the zero and one targets [0.0 0.3 1.0]

function test {
  mkdir -p $1
  cp $2/$3-adagrad-trace.netcdf $1
  ./IsotopeSep/isosep.py \
    --barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids \
    --summarydirs $4 \
    --posteriorpredictive \
    --zero $5 --one $6 \
    --outputdir $1 \
    --kmer $3 \
    --sampler adagrad
}

# copy over trained models for tests

#mkdir -p ./test-d2o-0-30
#mkdir -p ./test-d2o-0-30-100  # trained on 0-30 but tested on 0-100
#cp ./train-d2o-0-30/5-adagrad-trace.netcdf ./test-d2o-0-30

#./IsotopeSep/isosep.py \
#  --barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids \
#  --summarydirs ../_data/data2x \
#  --posteriorpredictive \
#  --zero 0.0 --one 0.3 \
#  --outputdir ./test-d2o-0-30 \
#  --kmer 5 \
#  --sampler adagrad

train ./train-d2o-0-30   5  ../_data/data1x  0.0  0.3
train ./train-d2o-0-100  5  ../_data/data1x  0.0  1.0

test ./test-d2o-0-30   ./train-d2o-0-30   5  ../_data/data2x  0.0  0.3
test ./test-d2o-0-100  ./train-d2o-0-100  5  ../_data/data2x  0.0  1.0
test ./test-d2o-0-30-100  ./train-d2o-0-30  5  ../_data/data2x  0.0  1.0


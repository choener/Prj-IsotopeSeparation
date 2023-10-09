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
      $6 \
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
    $6 \
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
    $7 \
    --summarydirs $4 \
    --posteriorpredictive \
    --zero $5 --one $6 \
    --outputdir $1 \
    --kmer $3 \
    --sampler adagrad
}

# # deuterium
# 
# train ./train-d2o-0-30   5  ../_data/data1x  0.0  0.3   "--barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids"
# train ./train-d2o-0-100  5  ../_data/data1x  0.0  1.0   "--barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids"
# 
# test ./test-d2o-0-30   ./train-d2o-0-30   5  ../_data/data2x  0.0  0.3   "--barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids"
# test ./test-d2o-0-100  ./train-d2o-0-100  5  ../_data/data2x  0.0  1.0   "--barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids"
# test ./test-d2o-0-30-100  ./train-d2o-0-30  5  ../_data/data2x  0.0  1.0   "--barcode 0 taubert-d2o/barcode14.ids --barcode 30 taubert-d2o/barcode15.ids --barcode 100 taubert-d2o/barcode16.ids"

# carbon

train ./train-carbon-0-100 5 ../_data/carbon/data/train 0.0 1.0   "--barcode 0 ../_data/carbon/barcodes/barcode20.ids --barcode 100 ../_data/carbon/barcodes/barcode21.ids"

test ./test-carbon-0-100 ./train-carbon-0-100 5 ../_data/carbon/data/test 0.0 1.0   "--barcode 0 ../_data/carbon/barcodes/barcode20.ids --barcode 100 ../_data/carbon/barcodes/barcode21.ids"

